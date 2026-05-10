"""
graph/event_graph.py — EventGraph

Async Neo4j driver wrapper. Manages all node and edge operations
for the Clary event causality graph.

Node types (from graph_schema.py):
  User, Session, Symptom, Trigger, Pattern, Mechanism

Edge types:
  HAS_SESSION, REPORTED_SYMPTOM, CONTAINS_TRIGGER, PRECEDES,
  SIMILAR_TO, CAUSED_BY, EXPLAINED_BY, EVIDENCE_FOR, DOWNSTREAM_OF

Key methods:
  - ingest_pipeline_context(ctx)  ← called by Orchestrator after each session
  - upsert_pattern(pattern)       ← called when pattern status changes
  - get_user_timeline(user_id)    ← called by UI timeline page
  - get_subgraph(user_id)         ← called by graph visualiser
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import neo4j
import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from config import get_settings
from graph import graph_queries as Q
from graph.graph_schema import (
    ALL_DDL,
    CausedByRel,
    EvidenceForRel,
    MechanismNode,
    PatternNode,
    PrecedesRel,
    RelationshipType,
    SessionNode,
    SymptomNode,
    TriggerNode,
    UserNode,
)
from schemas.event import HealthEvent
from schemas.pattern import PatternStatus, TemporalPattern
from schemas.session import PipelineContext
from schemas.user import UserProfile

logger = structlog.get_logger(__name__)
settings = get_settings()


class EventGraph:
    """
    Async Neo4j graph manager for Clary event data.

    Connection pooling is handled by the Neo4j driver internally.
    All public methods accept and return plain Python dicts/lists
    (no Neo4j Record objects leak outside this class).
    """

    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                database=settings.neo4j_database,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
            )
            await self._run_ddl()
        return self._driver

    async def _run_ddl(self) -> None:
        """Apply constraint and index DDL on first connection."""
        driver = self._driver
        assert driver is not None
        async with driver.session(database=settings.neo4j_database) as session:
            for stmt in ALL_DDL:
                try:
                    await session.run(stmt)
                except Exception as exc:
                    logger.warning("graph.ddl_warning", stmt=stmt[:60], error=str(exc))
        logger.info("graph.ddl_applied", statements=len(ALL_DDL))

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ── Primary pipeline integration ───────────────────────────────────────

    async def ingest_pipeline_context(self, ctx: PipelineContext) -> None:
        """
        Called by Orchestrator after each successful session.
        Persists all nodes and edges derived from the pipeline run.
        """
        driver = await self._get_driver()
        async with driver.session(database=settings.neo4j_database) as session:
            async with session.begin_transaction() as tx:
                try:
                    # 1. Upsert User node
                    await tx.run(
                        Q.UPSERT_USER,
                        user_id=ctx.user_id,
                        name="",
                        age=None,
                        location="",
                        created_at=datetime.utcnow().isoformat(),
                    )

                    # 2. Process each HealthEvent
                    for event in ctx.current_events:
                        await self._ingest_event(tx, event)

                    # 3. Create PRECEDES edges between consecutive sessions
                    if ctx.current_events:
                        await self._create_precedes_edges(tx, ctx)

                    # 4. Upsert validated patterns
                    for pattern in ctx.validated_patterns:
                        await self._upsert_pattern_subgraph(tx, pattern)

                    await tx.commit()
                    logger.info(
                        "graph.ingested",
                        session_id=ctx.session_id,
                        events=len(ctx.current_events),
                        patterns=len(ctx.validated_patterns),
                    )
                except Exception as exc:
                    await tx.rollback()
                    logger.error(
                        "graph.ingest_failed",
                        session_id=ctx.session_id,
                        error=str(exc),
                    )
                    raise

    async def _ingest_event(self, tx: Any, event: HealthEvent) -> None:
        """Upsert Session node + Symptom/Trigger nodes + linking edges."""
        # Session node
        await tx.run(
            Q.UPSERT_SESSION,
            session_id=event.session_id,
            user_id=event.user_id,
            timestamp_iso=event.timestamp.isoformat(),
            timestamp_epoch=event.timestamp_epoch,
            severity=event.severity.value,
            summary=event.raw_text[:200],
            symptoms=",".join(event.symptoms),
            triggers=",".join(event.triggers + event.behaviors),
        )

        # Link User → Session
        await tx.run(
            Q.LINK_USER_SESSION,
            user_id=event.user_id,
            session_id=event.session_id,
        )

        # Symptom nodes + edges
        for symptom in event.symptoms:
            sym_norm = symptom.lower().strip()
            if not sym_norm:
                continue
            await tx.run(
                Q.UPSERT_SYMPTOM,
                name=sym_norm,
                display_name=symptom,
                body_location=event.body_location.value,
            )
            await tx.run(
                Q.LINK_SESSION_SYMPTOM,
                session_id=event.session_id,
                symptom_name=sym_norm,
                severity=event.severity.value,
                timestamp_epoch=event.timestamp_epoch,
            )

        # Trigger nodes + edges
        for trigger in event.triggers + event.behaviors:
            trig_norm = trigger.lower().strip()
            if not trig_norm:
                continue
            await tx.run(
                Q.UPSERT_TRIGGER,
                name=trig_norm,
                display_name=trigger,
                trigger_type="behavior" if trigger in event.behaviors else "unknown",
            )
            await tx.run(
                Q.LINK_SESSION_TRIGGER,
                session_id=event.session_id,
                trigger_name=trig_norm,
                certainty="explicit",
            )

    async def _create_precedes_edges(
        self, tx: Any, ctx: PipelineContext
    ) -> None:
        """
        Create PRECEDES edges from any prior session to the current session.
        We link the latest prior session (closest in time).
        """
        if not ctx.current_events:
            return

        current_event = ctx.current_events[0]
        current_session_id = current_event.session_id
        current_epoch = current_event.timestamp_epoch

        # Find shared symptoms with prior sessions (from validated_patterns evidence)
        shared_symptoms: set[str] = set(current_event.symptoms)

        for pattern in ctx.validated_patterns:
            for ev in pattern.evidence:
                if ev.session_id == current_session_id:
                    continue
                # Link this evidence session → current session
                lag = (current_epoch - ev.occurred_at.timestamp()) / 86400.0
                if lag < 0:
                    continue
                await tx.run(
                    Q.LINK_SESSION_PRECEDES,
                    session_id_from=ev.session_id,
                    session_id_to=current_session_id,
                    lag_days=round(lag, 1),
                    same_symptom=pattern.symptom in shared_symptoms,
                    shared_symptoms=pattern.symptom,
                )

    async def _upsert_pattern_subgraph(
        self, tx: Any, pattern: TemporalPattern
    ) -> None:
        """Upsert Pattern node + Mechanism + causal edges + evidence links."""
        pattern_id_str = str(pattern.pattern_id)

        # Pattern node
        await tx.run(
            Q.UPSERT_PATTERN,
            pattern_id=pattern_id_str,
            user_id=pattern.user_id,
            symptom=pattern.symptom,
            trigger=pattern.trigger,
            status=pattern.status.value,
            confidence=pattern.confidence.value,
            occurrence_count=pattern.occurrence_count,
            lag_days_min=pattern.lag_days_min,
            lag_days_max=pattern.lag_days_max,
            first_detected_at=pattern.first_detected_at.isoformat(),
            last_confirmed_at=pattern.last_confirmed_at.isoformat(),
        )

        # Causal edge: Symptom -[CAUSED_BY]-> Trigger
        sym_norm = pattern.symptom.lower().strip()
        trig_norm = pattern.trigger.lower().strip()

        if sym_norm and trig_norm:
            # Ensure nodes exist
            await tx.run(
                Q.UPSERT_SYMPTOM, name=sym_norm,
                display_name=pattern.symptom, body_location=""
            )
            await tx.run(
                Q.UPSERT_TRIGGER, name=trig_norm,
                display_name=pattern.trigger, trigger_type=pattern.trigger_category or "unknown"
            )
            mean_lag = pattern.lag_days_mean or 0.0
            await tx.run(
                Q.LINK_SYMPTOM_CAUSED_BY,
                symptom_name=sym_norm,
                trigger_name=trig_norm,
                confidence=pattern.confidence.value,
                evidence_count=pattern.occurrence_count,
                mean_lag_days=round(mean_lag, 1),
            )

        # Mechanism node + EXPLAINED_BY edge
        if pattern.lag_registry_match:
            mech = pattern.lag_registry_match
            await tx.run(
                Q.UPSERT_MECHANISM,
                name=mech.mechanism_name,
                description=mech.description[:500],
                lag_min_days=mech.lag_min_days,
                lag_max_days=mech.lag_max_days,
            )
            await tx.run(
                Q.LINK_PATTERN_MECHANISM,
                pattern_id=pattern_id_str,
                mechanism_name=mech.mechanism_name,
                match_quality=mech.match_quality,
            )

        # Evidence edges: Session -[EVIDENCE_FOR]-> Pattern
        for i, ev in enumerate(pattern.evidence, 1):
            await tx.run(
                Q.LINK_SESSION_EVIDENCE_FOR_PATTERN,
                session_id=ev.session_id,
                pattern_id=pattern_id_str,
                occurrence_number=i,
                lag_days_actual=round(ev.lag_days, 1),
            )

        # Cascade / downstream edges
        await self._create_downstream_edges(tx, pattern)

    async def _create_downstream_edges(
        self, tx: Any, pattern: TemporalPattern
    ) -> None:
        """
        Create DOWNSTREAM_OF edges for cascade patterns.
        E.g., caloric_restriction → dizziness → brain_fog → hair_fall
        """
        # Only create if lag is >= 7 days (indicates downstream causal chain)
        if pattern.lag_days_min < 7:
            return

        sym_norm = pattern.symptom.lower().strip()
        trig_norm = pattern.trigger.lower().strip()

        if not sym_norm or not trig_norm:
            return

        # If trigger is itself a symptom (cascade), create DOWNSTREAM_OF
        await tx.run(
            """
            MATCH (upstream:Symptom {name: $trigger_as_symptom})
            MATCH (downstream:Symptom {name: $symptom})
            MERGE (downstream)-[r:DOWNSTREAM_OF]->(upstream)
            ON CREATE SET r.weeks_delay = $weeks, r.user_id = $user_id
            """,
            trigger_as_symptom=trig_norm,
            symptom=sym_norm,
            weeks=round(pattern.lag_days_min / 7, 1),
            user_id=pattern.user_id,
        )

    # ── Read API ───────────────────────────────────────────────────────────

    async def get_user_timeline(self, user_id: str) -> list[dict[str, Any]]:
        """Return all sessions for a user, sorted by timestamp."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(Q.GET_USER_TIMELINE, user_id=user_id)
            records = await result.data()
            logger.debug("graph.timeline_fetched", user_id=user_id, count=len(records))
            return records

    async def get_patterns_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Return all Pattern nodes for a user."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(Q.GET_PATTERNS_FOR_USER, user_id=user_id)
            return await result.data()

    async def get_pattern_evidence(self, pattern_id: str) -> list[dict[str, Any]]:
        """Return all sessions that are evidence for a pattern."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                Q.GET_PATTERN_EVIDENCE_SESSIONS, pattern_id=pattern_id
            )
            return await result.data()

    async def detect_cascades(self, user_id: str) -> list[dict[str, Any]]:
        """Find symptom cascade chains for a user."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(Q.DETECT_CASCADE_CHAINS, user_id=user_id)
            return await result.data()

    async def get_consistent_triggers(
        self,
        user_id: str,
        symptom: str,
        max_lag_days: int = 84,
    ) -> list[dict[str, Any]]:
        """Find triggers consistently preceding a symptom (for validation)."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                Q.GET_CONSISTENT_PRIOR_TRIGGERS,
                user_id=user_id,
                symptom=symptom.lower(),
                max_lag_days=max_lag_days,
            )
            return await result.data()

    async def get_subgraph_for_user(self, user_id: str) -> dict[str, Any]:
        """
        Fetch the full user subgraph as nodes + edges for visualisation.
        Returns a dict suitable for pyvis/networkx rendering.
        """
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(Q.GET_USER_GRAPH_SUBGRAPH, user_id=user_id)
            raw = await result.data()

        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        for row in raw:
            for key, val in row.items():
                if val is None:
                    continue
                if hasattr(val, "element_id"):
                    # It's a Neo4j Node or Relationship
                    node_id = val.element_id
                    if hasattr(val, "labels"):  # Node
                        label = list(val.labels)[0] if val.labels else "Unknown"
                        props = dict(val)
                        nodes[node_id] = {"id": node_id, "label": label, "props": props}
                    else:  # Relationship
                        edges.append({
                            "from": val.start_node.element_id if hasattr(val, "start_node") else "",
                            "to": val.end_node.element_id if hasattr(val, "end_node") else "",
                            "type": val.type if hasattr(val, "type") else "",
                            "props": dict(val),
                        })

        return {"nodes": list(nodes.values()), "edges": edges}

    # ── Health check ───────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            driver = await self._get_driver()
            async with driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as exc:
            logger.error("graph.ping_failed", error=str(exc))
            return False