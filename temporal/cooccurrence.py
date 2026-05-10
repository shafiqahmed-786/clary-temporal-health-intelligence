"""
temporal/cooccurrence.py — CoOccurrenceMatrix

Maintains a per-user sparse count matrix: symptom × trigger → count.
Also tracks the mean observed lag for each pair.

Used by PatternAgent to:
  1. Quickly identify high-frequency symptom-trigger pairs.
  2. Rank trigger candidates before handing to LLM analysis.
  3. Power the timeline visualizer.

In production this is persisted to PostgreSQL (via SQLAlchemy).
In this module we provide the in-memory implementation with a
serialisable state dict for persistence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CoOccurrenceCell:
    """Data for a single (symptom, trigger) pair."""

    symptom: str
    trigger: str
    count: int = 0
    total_lag_days: float = 0.0
    min_lag_days: float | None = None
    max_lag_days: float | None = None
    session_ids: list[str] = field(default_factory=list)

    @property
    def mean_lag_days(self) -> float | None:
        if self.count == 0:
            return None
        return self.total_lag_days / self.count

    def update(self, lag_days: float, session_id: str) -> None:
        self.count += 1
        self.total_lag_days += lag_days
        self.min_lag_days = (
            lag_days if self.min_lag_days is None else min(self.min_lag_days, lag_days)
        )
        self.max_lag_days = (
            lag_days if self.max_lag_days is None else max(self.max_lag_days, lag_days)
        )
        if session_id not in self.session_ids:
            self.session_ids.append(session_id)


class CoOccurrenceMatrix:
    """
    Sparse in-memory co-occurrence matrix for one user.

    Structure:
        _matrix[symptom][trigger] → CoOccurrenceCell
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._matrix: dict[str, dict[str, CoOccurrenceCell]] = defaultdict(dict)

    # ── Write ──────────────────────────────────────────────────────────────

    def record(
        self,
        symptom: str,
        trigger: str,
        lag_days: float,
        session_id: str,
    ) -> None:
        """Record one (symptom, trigger) observation with its lag."""
        sym = symptom.lower().strip()
        trig = trigger.lower().strip()

        if trig not in self._matrix[sym]:
            self._matrix[sym][trig] = CoOccurrenceCell(symptom=sym, trigger=trig)

        self._matrix[sym][trig].update(lag_days, session_id)

        logger.debug(
            "cooccurrence.recorded",
            user_id=self.user_id,
            symptom=sym,
            trigger=trig,
            lag_days=lag_days,
            new_count=self._matrix[sym][trig].count,
        )

    def record_batch(
        self,
        symptoms: list[str],
        triggers: list[str],
        lag_days: float,
        session_id: str,
    ) -> None:
        """Record all combinations of symptoms × triggers from one session."""
        for sym in symptoms:
            for trig in triggers:
                self.record(sym, trig, lag_days, session_id)

    # ── Read ───────────────────────────────────────────────────────────────

    def get_cell(self, symptom: str, trigger: str) -> CoOccurrenceCell | None:
        sym = symptom.lower().strip()
        trig = trigger.lower().strip()
        return self._matrix.get(sym, {}).get(trig)

    def get_triggers_for_symptom(
        self,
        symptom: str,
        min_count: int = 2,
        top_n: int = 10,
    ) -> list[CoOccurrenceCell]:
        """
        Return triggers for a symptom, ranked by count descending.
        Only returns cells with count ≥ min_count.
        """
        sym = symptom.lower().strip()
        cells = [
            cell for cell in self._matrix.get(sym, {}).values()
            if cell.count >= min_count
        ]
        cells.sort(key=lambda c: c.count, reverse=True)
        return cells[:top_n]

    def get_symptoms_for_trigger(
        self,
        trigger: str,
        min_count: int = 2,
    ) -> list[CoOccurrenceCell]:
        """Return all symptoms co-occurring with a trigger."""
        trig = trigger.lower().strip()
        result = []
        for sym_dict in self._matrix.values():
            if trig in sym_dict and sym_dict[trig].count >= min_count:
                result.append(sym_dict[trig])
        result.sort(key=lambda c: c.count, reverse=True)
        return result

    def get_top_pairs(self, min_count: int = 2, top_n: int = 20) -> list[CoOccurrenceCell]:
        """Return the globally highest-count (symptom, trigger) pairs."""
        all_cells: list[CoOccurrenceCell] = []
        for sym_dict in self._matrix.values():
            for cell in sym_dict.values():
                if cell.count >= min_count:
                    all_cells.append(cell)
        all_cells.sort(key=lambda c: c.count, reverse=True)
        return all_cells[:top_n]

    def has_pair(self, symptom: str, trigger: str, min_count: int = 1) -> bool:
        cell = self.get_cell(symptom, trigger)
        return cell is not None and cell.count >= min_count

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise for database persistence."""
        result: dict = {}
        for sym, trig_dict in self._matrix.items():
            result[sym] = {}
            for trig, cell in trig_dict.items():
                result[sym][trig] = {
                    "count": cell.count,
                    "total_lag_days": cell.total_lag_days,
                    "min_lag_days": cell.min_lag_days,
                    "max_lag_days": cell.max_lag_days,
                    "session_ids": cell.session_ids,
                }
        return result

    @classmethod
    def from_dict(cls, user_id: str, data: dict) -> "CoOccurrenceMatrix":
        """Restore from serialised dict."""
        mat = cls(user_id=user_id)
        for sym, trig_dict in data.items():
            for trig, vals in trig_dict.items():
                cell = CoOccurrenceCell(symptom=sym, trigger=trig)
                cell.count = vals["count"]
                cell.total_lag_days = vals["total_lag_days"]
                cell.min_lag_days = vals["min_lag_days"]
                cell.max_lag_days = vals["max_lag_days"]
                cell.session_ids = vals["session_ids"]
                mat._matrix[sym][trig] = cell
        return mat

    def summary(self) -> str:
        """Human-readable summary for debugging."""
        top = self.get_top_pairs(min_count=1, top_n=5)
        lines = [f"CoOccurrenceMatrix for user {self.user_id}:"]
        for cell in top:
            lines.append(
                f"  {cell.symptom} ← {cell.trigger}: "
                f"count={cell.count}, mean_lag={cell.mean_lag_days:.1f}d"
            )
        return "\n".join(lines)