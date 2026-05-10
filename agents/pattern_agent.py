"""
agents/pattern_agent.py — PatternDetectionAgent

Two-phase architecture:
  Phase 1 (algorithmic): EventTimeline + CoOccurrenceMatrix + HypothesisBuilder
    → fast, deterministic, no LLM tokens spent
    → produces CandidatePattern list with evidence pre-populated

  Phase 2 (LLM): GPT-4o receives the pre-structured candidates + full
    timeline + lag registry description
    → refines candidates, adds notes, catches patterns the algorithm missed
    → returns structured JSON with reasoning trace

Why two phases?
  The algorithm guarantees correctness properties (temporal ordering,
  minimum evidence count, consistency score). The LLM adds:
    - Natural language pattern naming
    - Nuanced lag window estimation
    - Detection of interaction effects the algorithm misses
    - Identification of cascade patterns (P8: caloric restriction → dizziness → fog → hair fall)
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

import structlog

from agents.base_agent import AgentError, BaseAgent
from config import get_settings
from memory.episodic_store import EpisodicStore
from schemas.event import HealthEvent, Severity
from schemas.pattern import (
    CandidatePattern,
    ConfidenceLevel,
    LagRegistryMatch,
    PatternEvidence,
    PatternStatus,
    TemporalPattern,
)
from schemas.session import PipelineContext
from temporal.cooccurrence import CoOccurrenceMatrix
from temporal.event_timeline import EventTimeline
from temporal.hypothesis_builder import build_candidates
from temporal.lag_detector import lag_registry

logger = structlog.get_logger(__name__)
settings = get_settings()


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are the Pattern Detection Agent for Clary, a health intelligence assistant.
Your sole job: analyse a user's health timeline and identify TEMPORAL HEALTH PATTERNS.

A temporal pattern = the SAME symptom recurring, with the SAME trigger consistently
preceding it within a biologically plausible lag window.

══ HARD RULES (never violate) ══
1. NEVER diagnose a medical condition.
2. NEVER assert a pattern with fewer than 2 confirmed occurrences.
3. Trigger MUST always precede symptom (positive lag_days). Reject if lag < 0.
4. Report ALL co-occurring variables for each occurrence — these are potential confounders.
5. Your output is consumed by a Skeptic Agent; be honest about uncertainty.

══ YOUR ANALYSIS PROCESS ══
Step 1: Scan the timeline. Group events by symptom keyword.
Step 2: For each symptom with ≥2 occurrences, look back within the lag window.
Step 3: Find what trigger/behaviour is CONSISTENTLY present before EVERY occurrence.
Step 4: Calculate mean lag_days for each (symptom, trigger) pair.
Step 5: Note whether a known biological lag window matches (see lag registry below).
Step 6: Report ALL patterns, even weak ones — the Skeptic will filter.
Step 7: Look for CASCADE patterns: symptom A → triggers symptom B weeks later.

══ KNOWN BIOLOGICAL LAG WINDOWS (match these) ══
{lag_registry}

══ OUTPUT FORMAT ══
Return ONLY a valid JSON object. No markdown, no preamble.
""".strip()

_USER_TEMPLATE = """
{context_window}

══ ALGORITHMICALLY DETECTED CANDIDATES (pre-validated) ══
{algo_candidates}

══ TASK ══
Review the timeline and algorithmic candidates. Then:
1. Confirm, refine, or reject each algorithmic candidate.
2. Add any NEW patterns the algorithm missed (especially cascade patterns).
3. For each pattern, list ALL occurrences with exact dates and lag_days.
4. Assign a consistency_score (0.0-1.0): fraction of symptom occurrences where trigger was present.

Return JSON matching this schema exactly:
{{
  "patterns": [
    {{
      "symptom": "string — normalised symptom name",
      "trigger": "string — normalised trigger name",
      "title": "string — short human-readable pattern name",
      "trigger_category": "food|behavior|stress|environmental|unknown",
      "occurrences": [
        {{
          "session_id": "string",
          "symptom_date": "YYYY-MM-DD",
          "trigger_date": "YYYY-MM-DD or null",
          "lag_days": 0.0,
          "symptom_severity": "none|mild|moderate|severe|unknown",
          "co_occurring_variables": ["list", "of", "other", "triggers"],
          "raw_excerpt": "brief quote from session ≤30 words"
        }}
      ],
      "lag_days_min": 0,
      "lag_days_max": 0,
      "consistency_score": 0.0,
      "registry_mechanism": "mechanism name or null",
      "analysis_notes": "string"
    }}
  ],
  "cascade_patterns": [
    {{
      "description": "string — describe the chain",
      "chain": ["symptom_A", "symptom_B", "symptom_C"],
      "trigger": "string",
      "total_lag_weeks": 0
    }}
  ],
  "reasoning": "string — step-by-step reasoning trace"
}}
"""


class PatternDetectionAgent(BaseAgent):
    """
    Detects temporal health patterns across a user's session history.

    Inputs (from PipelineContext):
      - user_id
      - context_window_text (built by ContextBuilder)
      - current_events (this session's parsed events)

    Outputs (written to PipelineContext):
      - candidate_patterns: list[CandidatePattern]
    """

    AGENT_NAME = "pattern_detection_agent"

    def __init__(self, episodic_store: EpisodicStore) -> None:
        super().__init__()
        self._episodic = episodic_store

    # ── Main entry point ───────────────────────────────────────────────────

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        self._log.info(
            "pattern_agent.start",
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            current_events=len(ctx.current_events),
        )

        # Need at least 1 prior session to detect patterns
        event_count = await self._episodic.event_count(ctx.user_id)
        if event_count < settings.min_evidence_count:
            self._log.info(
                "pattern_agent.insufficient_history",
                user_id=ctx.user_id,
                event_count=event_count,
                needed=settings.min_evidence_count,
            )
            ctx.candidate_patterns = []
            return ctx

        # ── Phase 1: Algorithmic candidate generation ──────────────────────
        timeline, matrix, algo_candidates = await self._run_algorithmic_phase(ctx)

        # ── Phase 2: LLM refinement and cascade detection ──────────────────
        llm_candidates = await self._run_llm_phase(
            ctx=ctx,
            timeline=timeline,
            algo_candidates=algo_candidates,
        )

        # ── Merge phases ───────────────────────────────────────────────────
        final_candidates = self._merge_candidates(
            algo=algo_candidates,
            llm=llm_candidates,
            user_id=ctx.user_id,
        )

        ctx.candidate_patterns = final_candidates
        self._log.info(
            "pattern_agent.done",
            user_id=ctx.user_id,
            candidates=len(final_candidates),
        )
        return ctx

    # ── Phase 1: Algorithmic ───────────────────────────────────────────────

    async def _run_algorithmic_phase(
        self,
        ctx: PipelineContext,
    ) -> tuple[EventTimeline, CoOccurrenceMatrix, list[CandidatePattern]]:
        """
        Build EventTimeline + CoOccurrenceMatrix from episodic store,
        then run hypothesis builder to get algorithmic candidates.
        """
        all_events_raw = await self._episodic.get_all_events(ctx.user_id)

        # Convert raw dicts back to HealthEvent-like objects for timeline
        events = self._raw_to_health_events(all_events_raw, ctx.user_id)

        # Add current session's events (not yet stored)
        events.extend(ctx.current_events)

        # Build timeline
        timeline = EventTimeline(events=events, user_id=ctx.user_id)

        # Build co-occurrence matrix
        matrix = CoOccurrenceMatrix(user_id=ctx.user_id)
        for ev in events:
            for sym in ev.symptoms:
                for trig in ev.triggers + ev.behaviors:
                    # Use 0 as lag placeholder for matrix construction
                    matrix.record(symptom=sym, trigger=trig, lag_days=0.0, session_id=ev.session_id)

        algo_candidates = build_candidates(
            timeline=timeline,
            matrix=matrix,
            min_evidence=settings.min_evidence_count,
        )

        self._log.info(
            "pattern_agent.algorithmic_phase",
            user_id=ctx.user_id,
            algo_candidates=len(algo_candidates),
        )
        return timeline, matrix, algo_candidates

    # ── Phase 2: LLM ──────────────────────────────────────────────────────

    async def _run_llm_phase(
        self,
        ctx: PipelineContext,
        timeline: EventTimeline,
        algo_candidates: list[CandidatePattern],
    ) -> list[dict]:
        """
        Send timeline + algorithmic candidates to GPT-4o for refinement.
        Returns parsed list of LLM pattern dicts.
        """
        system_prompt = _SYSTEM_PROMPT.format(
            lag_registry=lag_registry.describe_all()
        )

        algo_text = self._format_algo_candidates(algo_candidates)
        user_content = _USER_TEMPLATE.format(
            context_window=ctx.context_window_text,
            algo_candidates=algo_text if algo_text else "None detected algorithmically.",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            raw, trace = await self._llm_call(
                messages=messages,
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                temperature=0.1,
            )
        except AgentError as exc:
            self._log.error("pattern_agent.llm_failed", error=str(exc))
            ctx.errors.append(f"PatternAgent LLM failed: {exc}")
            return []

        self._record_trace(ctx, trace, f"Pattern LLM: {trace.usage.total_tokens} tokens")

        parsed = self._parse_json(raw, context="PatternAgent")
        patterns = parsed.get("patterns", [])
        reasoning = parsed.get("reasoning", "")

        self._log.info(
            "pattern_agent.llm_phase",
            user_id=ctx.user_id,
            llm_patterns=len(patterns),
            reasoning_chars=len(reasoning),
        )
        return patterns

    # ── Merge ──────────────────────────────────────────────────────────────

    def _merge_candidates(
        self,
        algo: list[CandidatePattern],
        llm: list[dict],
        user_id: str,
    ) -> list[CandidatePattern]:
        """
        Merge algorithmic and LLM candidates.
        Strategy:
          - Algorithmic candidates with LLM confirmation → keep, enrich with LLM notes
          - LLM-only candidates with N≥2 → keep as new candidates
          - Algorithmic candidates with LLM rejection → lower consistency score
          - LLM-only with N<2 → discard
        """
        seen: dict[tuple[str, str], CandidatePattern] = {}

        # Index algorithmic candidates
        for c in algo:
            key = (c.symptom.lower(), c.trigger.lower())
            seen[key] = c

        # Process LLM candidates
        for llm_pat in llm:
            symptom = (llm_pat.get("symptom") or "").lower().strip()
            trigger = (llm_pat.get("trigger") or "").lower().strip()
            if not symptom or not trigger:
                continue

            occurrences = llm_pat.get("occurrences", [])
            if len(occurrences) < settings.min_evidence_count:
                continue

            key = (symptom, trigger)

            if key in seen:
                # Enrich existing algorithmic candidate with LLM analysis notes
                seen[key].raw_analysis_notes += (
                    f"\nLLM analysis: {llm_pat.get('analysis_notes', '')}"
                )
                continue

            # New LLM-only candidate — build evidence list
            evidence = self._build_evidence_from_llm_occurrences(
                occurrences=occurrences,
                user_id=user_id,
            )
            if len(evidence) < settings.min_evidence_count:
                continue

            # Check registry for this new pair
            mean_lag = (
                sum(e.lag_days for e in evidence) / len(evidence) if evidence else None
            )
            registry_match: LagRegistryMatch | None = None
            if llm_pat.get("registry_mechanism"):
                registry_match = lag_registry.lookup(symptom, trigger, mean_lag)

            candidate = CandidatePattern(
                user_id=user_id,
                symptom=symptom,
                trigger=trigger,
                evidence=evidence,
                consistency_score=float(llm_pat.get("consistency_score", 0.5)),
                lag_registry_match=registry_match,
                raw_analysis_notes=(
                    f"LLM-detected. Title: {llm_pat.get('title', '')}. "
                    f"{llm_pat.get('analysis_notes', '')}"
                ),
            )
            seen[key] = candidate
            self._log.info(
                "pattern_agent.llm_only_candidate",
                user_id=user_id,
                symptom=symptom,
                trigger=trigger,
            )

        result = list(seen.values())
        result.sort(
            key=lambda c: (c.lag_registry_match is not None, c.occurrence_count, c.consistency_score),
            reverse=True,
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _raw_to_health_events(
        raw_events: list[dict], user_id: str
    ) -> list[HealthEvent]:
        """Convert ChromaDB get() dicts back to minimal HealthEvent objects."""
        events: list[HealthEvent] = []
        for raw in raw_events:
            meta = raw.get("metadata", {})
            epoch = meta.get("timestamp_epoch")
            if not epoch:
                continue
            ts = datetime.fromtimestamp(float(epoch))
            symptoms = [s.strip() for s in (meta.get("symptoms") or "").split(",") if s.strip()]
            triggers = [t.strip() for t in (meta.get("triggers") or "").split(",") if t.strip()]
            sev_str = meta.get("severity", "unknown")

            try:
                sev = Severity(sev_str)
            except ValueError:
                sev = Severity.UNKNOWN

            ev = HealthEvent(
                user_id=user_id,
                session_id=meta.get("session_id", "unknown"),
                timestamp=ts,
                symptoms=symptoms,
                triggers=triggers,
                severity=sev,
                raw_text=raw.get("document", ""),
            )
            events.append(ev)
        return events

    @staticmethod
    def _format_algo_candidates(candidates: list[CandidatePattern]) -> str:
        if not candidates:
            return "None."
        lines = []
        for c in candidates:
            lines.append(
                f"  • {c.symptom} ← {c.trigger} "
                f"[N={c.occurrence_count}, consistency={c.consistency_score:.2f}"
                + (f", registry={c.lag_registry_match.mechanism_name}" if c.lag_registry_match else "")
                + "]"
            )
            for ev in c.evidence[:5]:
                lines.append(
                    f"    - {ev.occurred_at.strftime('%Y-%m-%d')} lag={ev.lag_days:.1f}d "
                    f"confounders={ev.co_occurring_variables[:3]}"
                )
        return "\n".join(lines)

    @staticmethod
    def _build_evidence_from_llm_occurrences(
        occurrences: list[dict],
        user_id: str,
    ) -> list[PatternEvidence]:
        """Build PatternEvidence objects from LLM occurrence dicts."""
        evidence: list[PatternEvidence] = []
        from uuid import uuid4
        for occ in occurrences:
            try:
                sym_date = datetime.fromisoformat(occ.get("symptom_date", "2000-01-01"))
            except (ValueError, TypeError):
                continue

            lag = float(occ.get("lag_days", 0))
            if lag < 0:
                continue  # Enforce temporal ordering

            trig_date_str = occ.get("trigger_date")
            trig_date = None
            if trig_date_str:
                try:
                    trig_date = datetime.fromisoformat(trig_date_str)
                except (ValueError, TypeError):
                    pass

            ev = PatternEvidence(
                session_id=occ.get("session_id", "unknown"),
                event_id=uuid4(),
                occurred_at=sym_date,
                trigger_present=True,
                trigger_observed_at=trig_date,
                lag_days=lag,
                symptom_severity=occ.get("symptom_severity", "unknown"),
                co_occurring_variables=occ.get("co_occurring_variables", [])[:10],
                raw_excerpt=occ.get("raw_excerpt", "")[:200],
            )
            evidence.append(ev)
        return evidence