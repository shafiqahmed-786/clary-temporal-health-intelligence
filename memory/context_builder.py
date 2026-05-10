"""
memory/context_builder.py — ContextBuilder

Assembles the LLM context window for each agent invocation.
Pulls from all three memory tiers and respects token budgets.

Context block order (strict, mirrors architecture spec):
  1. User profile          (~100 tokens)
  2. Rolling summary       (~350 tokens)   [sessions >30 days]
  3. Recent sessions       (~900 tokens)   [last 10 verbatim]
  4. Semantically relevant (~400 tokens)   [top-3 similarity]
  5. Confirmed patterns    (~250 tokens)
  6. Current message       (~250 tokens)
  Total budget: ~2250 tokens (leaves room for system prompt + response)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import structlog

from config import get_settings
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.summary_compressor import SummaryCompressor
from schemas.pattern import TemporalPattern
from schemas.user import UserProfile

logger = structlog.get_logger(__name__)
settings = get_settings()


def _session_to_text(sess: dict[str, Any], include_metadata: bool = True) -> str:
    """Format a retrieved session dict into a compact text block."""
    meta = sess.get("metadata", {})
    doc = sess.get("document", "")
    lines = []
    if include_metadata:
        date = ""
        epoch = meta.get("timestamp_epoch")
        if epoch:
            date = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")
        session_id = meta.get("session_id", "?")
        severity = meta.get("severity", "unknown")
        symptoms = meta.get("symptoms", "")
        triggers = meta.get("triggers", "")
        lines.append(
            f"[{date}] session={session_id} | sev={severity} | "
            f"symptoms=[{symptoms}] | triggers=[{triggers}]"
        )
    lines.append(f"  {doc}")
    return "\n".join(lines)


def _pattern_to_text(pattern: TemporalPattern) -> str:
    """Compact pattern summary for context window."""
    lag_str = f"{pattern.lag_days_min}–{pattern.lag_days_max}d"
    evidence_dates = [
        e.occurred_at.strftime("%Y-%m-%d") for e in pattern.evidence[:5]
    ]
    return (
        f"PATTERN ({pattern.status.value.upper()}, "
        f"confidence={pattern.confidence.value}): "
        f"{pattern.trigger} → {pattern.symptom} "
        f"[lag={lag_str}] "
        f"[N={pattern.occurrence_count}] "
        f"[dates={', '.join(evidence_dates)}]"
        + (f" [mechanism={pattern.lag_registry_match.mechanism_name}]"
           if pattern.lag_registry_match else "")
    )


class ContextBuilder:
    """
    Assembles the final context string injected into every agent LLM call.

    Usage:
        ctx = await builder.build(
            user_id=...,
            current_message=...,
            profile=...,
            confirmed_patterns=...,
        )
    """

    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        compressor: SummaryCompressor,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._compressor = compressor

    async def build(
        self,
        user_id: str,
        current_message: str,
        profile: UserProfile,
        confirmed_patterns: list[TemporalPattern] | None = None,
        include_semantic: bool = True,
    ) -> str:
        """
        Assemble the full context window string.
        Returns a formatted multi-block string suitable for LLM injection.
        """
        blocks: list[str] = []

        # ── Block 1: User profile ──────────────────────────────────────────
        profile_block = self._build_profile_block(profile)
        blocks.append(profile_block)

        # ── Block 2: Rolling summary (sessions > 30 days old) ─────────────
        summary = await self._compressor.get_or_build_summary(user_id)
        if summary:
            blocks.append(f"## HEALTH HISTORY SUMMARY (older sessions)\n{summary}")

        # ── Block 3: Recent sessions verbatim (last N) ────────────────────
        recent = await self._episodic.get_recent_events(
            user_id, n=settings.episodic_recent_n
        )
        if recent:
            recent_text = "\n".join(
                _session_to_text(s) for s in recent
            )
            blocks.append(f"## RECENT SESSIONS (last {len(recent)})\n{recent_text}")

        # ── Block 4: Semantically similar past sessions ────────────────────
        if include_semantic and current_message:
            semantic_hits = await self._episodic.semantic_search(
                user_id=user_id,
                query=current_message,
                top_k=settings.episodic_semantic_top_k,
            )
            if semantic_hits:
                sem_text = "\n".join(
                    _session_to_text(s) for s in semantic_hits
                )
                blocks.append(
                    f"## MOST RELEVANT PAST SESSIONS (semantic match)\n{sem_text}"
                )

        # ── Block 5: Confirmed patterns ────────────────────────────────────
        patterns = confirmed_patterns or []
        if patterns:
            pat_text = "\n".join(_pattern_to_text(p) for p in patterns)
            blocks.append(f"## CONFIRMED HEALTH PATTERNS\n{pat_text}")
        else:
            blocks.append("## CONFIRMED HEALTH PATTERNS\nNone confirmed yet.")

        # ── Block 6: Current message ───────────────────────────────────────
        blocks.append(f"## CURRENT USER MESSAGE\n{current_message}")

        context = "\n\n---\n\n".join(blocks)

        logger.info(
            "context_builder.assembled",
            user_id=user_id,
            blocks=len(blocks),
            approx_chars=len(context),
        )
        return context

    async def build_pattern_analysis_context(
        self,
        user_id: str,
        current_message: str,
        profile: UserProfile,
    ) -> str:
        """
        Extended context for PatternAgent — includes full event timeline
        and lag registry description in addition to standard blocks.
        """
        base = await self.build(
            user_id=user_id,
            current_message=current_message,
            profile=profile,
            include_semantic=True,
        )

        # Append full event history as structured timeline
        all_events = await self._episodic.get_all_events(user_id)
        if all_events:
            timeline_lines = ["## FULL CHRONOLOGICAL EVENT TIMELINE"]
            for ev in all_events:
                meta = ev.get("metadata", {})
                epoch = meta.get("timestamp_epoch", 0)
                date = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d") if epoch else "?"
                symptoms = meta.get("symptoms", "—")
                triggers = meta.get("triggers", "—")
                timeline_lines.append(
                    f"  {date} | session={meta.get('session_id', '?')} | "
                    f"symptoms=[{symptoms}] triggers=[{triggers}] "
                    f"sev={meta.get('severity', '?')}"
                )
            base += "\n\n---\n\n" + "\n".join(timeline_lines)

        return base

    @staticmethod
    def _build_profile_block(profile: UserProfile) -> str:
        lines = ["## USER PROFILE"]
        lines.append(f"Name: {profile.name or 'Unknown'}, Age: {profile.age or 'Unknown'}")
        if profile.known_conditions:
            lines.append(f"Known conditions: {', '.join(profile.known_conditions)}")
        if profile.known_allergies:
            lines.append(f"Allergies: {', '.join(profile.known_allergies)}")
        if profile.medications:
            lines.append(f"Medications: {', '.join(profile.medications)}")
        if profile.onboarding_notes:
            lines.append(f"Notes: {profile.onboarding_notes}")
        return "\n".join(lines)