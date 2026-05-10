"""
temporal/variable_isolator.py — VariableIsolator

Implements the variable isolation logic needed to solve patterns like P6/P7
in the dataset (sleep deprivation vs stress as the true driver of dysmenorrhea).

Core algorithm:
  Given N occurrences of a (symptom, outcome) pair, each with a set of
  co-occurring variables, find the MINIMAL SET of variables that are:
    1. Present in ALL occurrences (necessary condition)
    2. NOT equally explainable by an alternative variable

If stress is present in Cycles 1 & 2 but absent in Cycle 3 (where symptoms
still occur), stress is eliminated. Sleep deprivation is present in all three
→ identified as the consistent driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from schemas.pattern import PatternEvidence

logger = structlog.get_logger(__name__)


@dataclass
class IsolationResult:
    """Result of variable isolation across N occurrences."""

    consistent_variables: list[str]
    """Variables present in ALL occurrences — necessary conditions."""

    eliminated_variables: list[str]
    """Variables present in some but not all occurrences — rejected."""

    isolation_possible: bool
    """True if at least one consistent variable found."""

    occurrence_count: int

    evidence_matrix: list[dict[str, bool]]
    """Each dict: {variable: present_in_occurrence} — for transparency."""

    reasoning: str
    """Human-readable explanation of the isolation logic."""


class VariableIsolator:
    """
    Isolates consistent causal variables from confounders across
    multiple occurrences of a symptom pattern.

    Usage:
        isolator = VariableIsolator()
        result = isolator.isolate(evidence_list)
    """

    def isolate(
        self,
        evidence: list[PatternEvidence],
        candidate_trigger: str | None = None,
    ) -> IsolationResult:
        """
        Run variable isolation across all evidence sessions.

        Args:
            evidence: list of PatternEvidence, each with co_occurring_variables
            candidate_trigger: if provided, always tested first

        Returns:
            IsolationResult with consistent and eliminated variables
        """
        if not evidence:
            return IsolationResult(
                consistent_variables=[],
                eliminated_variables=[],
                isolation_possible=False,
                occurrence_count=0,
                evidence_matrix=[],
                reasoning="No evidence provided.",
            )

        # Build universe of all variables seen across all occurrences
        all_vars: set[str] = set()
        for ev in evidence:
            for v in ev.co_occurring_variables:
                all_vars.add(v.lower().strip())
        if candidate_trigger:
            all_vars.add(candidate_trigger.lower().strip())

        n = len(evidence)

        # Build presence matrix: for each variable, is it in each occurrence?
        presence: dict[str, list[bool]] = {v: [] for v in all_vars}
        evidence_matrix: list[dict[str, bool]] = []

        for ev in evidence:
            ev_vars = {v.lower().strip() for v in ev.co_occurring_variables}
            if candidate_trigger:
                ev_vars.add(candidate_trigger.lower().strip())
            row: dict[str, bool] = {}
            for var in all_vars:
                present = var in ev_vars
                presence[var].append(present)
                row[var] = present
            evidence_matrix.append(row)

        # Classify each variable
        consistent: list[str] = []
        eliminated: list[str] = []

        for var, presences in presence.items():
            if all(presences):
                consistent.append(var)
            else:
                eliminated.append(var)

        # Build reasoning trace
        lines = [f"Variable isolation across {n} occurrences:"]
        for var in sorted(all_vars):
            hits = sum(1 for p in presence[var] if p)
            status = "CONSISTENT ✓" if var in consistent else f"ELIMINATED (present {hits}/{n})"
            lines.append(f"  • {var}: {status}")

        if consistent:
            lines.append(
                f"\nConclusion: [{', '.join(consistent)}] present in ALL {n} occurrences "
                f"→ candidate causal factors."
            )
            if eliminated:
                lines.append(
                    f"Eliminated: [{', '.join(eliminated)}] absent from ≥1 occurrence "
                    f"→ not consistently necessary."
                )
        else:
            lines.append(
                "\nConclusion: No single variable consistent across all occurrences. "
                "Pattern may involve interaction effects or stochastic triggers."
            )

        reasoning = "\n".join(lines)
        logger.info(
            "variable_isolator.result",
            n_occurrences=n,
            consistent=consistent,
            eliminated=eliminated,
        )

        return IsolationResult(
            consistent_variables=consistent,
            eliminated_variables=eliminated,
            isolation_possible=len(consistent) > 0,
            occurrence_count=n,
            evidence_matrix=evidence_matrix,
            reasoning=reasoning,
        )

    def check_dose_response(
        self,
        evidence: list[PatternEvidence],
        trigger_intensity_key: str | None = None,
    ) -> bool:
        """
        Heuristic check: does higher trigger presence correlate with
        higher symptom severity across occurrences?

        Without numeric intensity data, we use severity string ordering.
        Returns True if a monotonic trend is detectable.
        """
        severity_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3, "unknown": -1}

        severities = [severity_order.get(ev.symptom_severity.lower(), -1) for ev in evidence]
        known = [(i, s) for i, s in enumerate(severities) if s >= 0]

        if len(known) < 3:
            return False

        # Check if sorted by occurrence index there's an upward trend
        scores = [s for _, s in known]
        ascending = sum(1 for i in range(len(scores) - 1) if scores[i] <= scores[i + 1])
        trend_ratio = ascending / (len(scores) - 1)

        result = trend_ratio >= 0.67
        logger.debug("variable_isolator.dose_response", detected=result, trend_ratio=trend_ratio)
        return result

    def check_removal_reversal(
        self,
        evidence: list[PatternEvidence],
        trigger: str,
        improvement_sessions: list[dict] | None = None,
    ) -> bool:
        """
        Check if the symptom improved when the trigger was absent.
        Looks for sessions where the trigger was NOT present and
        the outcome was better. Requires improvement_sessions data.
        """
        if not improvement_sessions:
            return False

        trig_norm = trigger.lower().strip()
        reversals = 0
        for sess in improvement_sessions:
            triggers_in_sess = [t.lower() for t in sess.get("triggers", [])]
            is_improvement = sess.get("is_improvement", False)
            trigger_absent = not any(trig_norm in t for t in triggers_in_sess)
            if is_improvement and trigger_absent:
                reversals += 1

        result = reversals >= 1
        logger.debug(
            "variable_isolator.removal_reversal", detected=result, reversals=reversals
        )
        return result