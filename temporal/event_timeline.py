"""
temporal/event_timeline.py — EventTimeline

Builds a chronological chain of HealthEvents for a user and provides
temporal query methods used by the Pattern Detection Agent.

Key capabilities:
  - get_events_in_window(start, end) → time-range filter
  - get_lookback_window(event, days) → events N days before an event
  - group_by_symptom() → {symptom: [events]}
  - group_by_trigger() → {trigger: [events]}
  - compute_lag(trigger_event, symptom_event) → float (days)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator

import structlog

from schemas.event import HealthEvent

logger = structlog.get_logger(__name__)


class EventTimeline:
    """
    Ordered, queryable sequence of HealthEvents for a single user.

    Events are maintained in ascending chronological order (oldest first).
    All query methods return new lists — the timeline itself is immutable
    after construction.
    """

    def __init__(self, events: list[HealthEvent], user_id: str) -> None:
        self.user_id = user_id
        self._events: list[HealthEvent] = sorted(
            events, key=lambda e: e.timestamp
        )
        logger.info(
            "timeline.built",
            user_id=user_id,
            event_count=len(self._events),
            span_days=self._span_days(),
        )

    # ── Core access ───────────────────────────────────────────────────────

    @property
    def events(self) -> list[HealthEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[HealthEvent]:
        return iter(self._events)

    def _span_days(self) -> float:
        if len(self._events) < 2:
            return 0.0
        return (self._events[-1].timestamp - self._events[0].timestamp).total_seconds() / 86400

    # ── Temporal queries ──────────────────────────────────────────────────

    def get_events_in_window(
        self,
        start: datetime,
        end: datetime,
        inclusive: bool = True,
    ) -> list[HealthEvent]:
        """Return events whose timestamp falls within [start, end]."""
        result = []
        for e in self._events:
            if inclusive:
                if start <= e.timestamp <= end:
                    result.append(e)
            else:
                if start < e.timestamp < end:
                    result.append(e)
        return result

    def get_lookback_window(
        self,
        anchor_event: HealthEvent,
        days: int,
        exclude_self: bool = True,
    ) -> list[HealthEvent]:
        """
        Return all events that occurred in the `days` before anchor_event.

        This is the fundamental operation for detecting lagged causality:
        given a symptom event, what happened in the preceding N days?
        """
        end = anchor_event.timestamp
        start = end - timedelta(days=days)
        candidates = self.get_events_in_window(start, end)
        if exclude_self:
            candidates = [e for e in candidates if e.event_id != anchor_event.event_id]
        return candidates

    def get_events_after(self, anchor: datetime, days: int) -> list[HealthEvent]:
        """Return events occurring up to `days` after anchor."""
        end = anchor + timedelta(days=days)
        return self.get_events_in_window(anchor, end)

    # ── Grouping ──────────────────────────────────────────────────────────

    def group_by_symptom(self) -> dict[str, list[HealthEvent]]:
        """
        Return {normalised_symptom: [events]} for all symptom events.
        Multiple symptoms per event are expanded.
        """
        out: dict[str, list[HealthEvent]] = {}
        for e in self._events:
            for sym in e.symptoms:
                key = sym.lower().strip()
                out.setdefault(key, []).append(e)
        return out

    def group_by_trigger(self) -> dict[str, list[HealthEvent]]:
        """Return {normalised_trigger: [events]}."""
        out: dict[str, list[HealthEvent]] = {}
        for e in self._events:
            for trig in e.triggers:
                key = trig.lower().strip()
                out.setdefault(key, []).append(e)
        return out

    def get_symptom_occurrences(self, symptom: str, min_count: int = 2) -> list[HealthEvent]:
        """
        Return events containing the given symptom keyword.
        Only returns results if occurrence count meets min_count threshold.
        """
        norm = symptom.lower().strip()
        hits = [
            e for e in self._events
            if any(norm in s.lower() for s in e.symptoms)
        ]
        if len(hits) < min_count:
            return []
        return hits

    def get_trigger_occurrences(self, trigger: str) -> list[HealthEvent]:
        """Return events containing the given trigger keyword."""
        norm = trigger.lower().strip()
        return [
            e for e in self._events
            if any(norm in t.lower() for t in e.triggers + e.behaviors)
        ]

    # ── Lag computation ───────────────────────────────────────────────────

    @staticmethod
    def compute_lag_days(trigger_event: HealthEvent, symptom_event: HealthEvent) -> float:
        """
        Compute days from trigger_event to symptom_event.
        Positive → trigger precedes symptom (correct temporal ordering).
        Negative → symptom precedes trigger (violation).
        """
        delta = symptom_event.timestamp - trigger_event.timestamp
        return delta.total_seconds() / 86400

    # ── Pattern support ───────────────────────────────────────────────────

    def find_consistent_prior_triggers(
        self,
        symptom_events: list[HealthEvent],
        lookback_days: int,
    ) -> dict[str, list[tuple[HealthEvent, float]]]:
        """
        For each symptom occurrence, collect the triggers present in the
        lookback window. Then return only triggers that appeared before
        EVERY symptom occurrence (consistency = 1.0).

        Returns:
            {trigger_name: [(trigger_event, lag_days), ...]}
            Only triggers with hits in ALL symptom occurrences are included.
        """
        if not symptom_events:
            return {}

        # For each symptom occurrence, build set of triggers seen in lookback
        per_occurrence: list[dict[str, list[tuple[HealthEvent, float]]]] = []

        for sym_ev in symptom_events:
            prior = self.get_lookback_window(sym_ev, lookback_days)
            trigger_map: dict[str, list[tuple[HealthEvent, float]]] = {}
            for p in prior:
                for trig in p.triggers + p.behaviors:
                    key = trig.lower().strip()
                    lag = self.compute_lag_days(p, sym_ev)
                    if lag >= 0:  # enforce temporal ordering
                        trigger_map.setdefault(key, []).append((p, lag))
            per_occurrence.append(trigger_map)

        if not per_occurrence:
            return {}

        # Intersection: triggers present in all occurrences
        universal_triggers = set(per_occurrence[0].keys())
        for occ in per_occurrence[1:]:
            universal_triggers &= set(occ.keys())

        logger.debug(
            "timeline.consistent_triggers",
            symptom_count=len(symptom_events),
            universal_trigger_count=len(universal_triggers),
            triggers=list(universal_triggers),
        )

        # Build result: each universal trigger with all its (event, lag) pairs
        result: dict[str, list[tuple[HealthEvent, float]]] = {}
        for trig in universal_triggers:
            all_pairs: list[tuple[HealthEvent, float]] = []
            for occ in per_occurrence:
                all_pairs.extend(occ[trig])
            result[trig] = all_pairs

        return result

    def to_prompt_text(self, max_events: int = 30) -> str:
        """
        Serialize timeline to a compact LLM-readable string.
        Most recent events last, truncated to max_events.
        """
        events = self._events[-max_events:]
        lines = [f"Health Timeline for user {self.user_id} ({len(events)} events shown):"]
        for e in events:
            date_str = e.timestamp.strftime("%Y-%m-%d")
            syms = ", ".join(e.symptoms) if e.symptoms else "—"
            trigs = ", ".join(e.triggers + e.behaviors) if (e.triggers or e.behaviors) else "—"
            sev = e.severity.value
            lines.append(
                f"  [{date_str}] session={e.session_id} | "
                f"symptoms=[{syms}] | triggers=[{trigs}] | severity={sev}"
            )
        return "\n".join(lines)