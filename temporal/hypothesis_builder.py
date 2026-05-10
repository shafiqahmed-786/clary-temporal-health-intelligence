"""
temporal/hypothesis_builder.py — HypothesisBuilder

Converts raw EventTimeline analysis into structured CandidatePattern objects
ready for Skeptic review. This module bridges the purely algorithmic
EventTimeline/CoOccurrenceMatrix layer and the LLM-powered PatternAgent.

It runs BEFORE the LLM call to pre-filter and structure the evidence,
reducing hallucination risk by grounding the prompt in validated signals.
"""

from __future__ import annotations

import structlog

from config import get_settings
from schemas.event import HealthEvent
from schemas.pattern import CandidatePattern, LagRegistryMatch, PatternEvidence
from temporal.cooccurrence import CoOccurrenceMatrix
from temporal.event_timeline import EventTimeline
from temporal.lag_detector import lag_registry
from temporal.variable_isolator import VariableIsolator

logger = structlog.get_logger(__name__)
settings = get_settings()

_isolator = VariableIsolator()


def _build_evidence(
    symptom_events: list[HealthEvent],
    trigger: str,
    timeline: EventTimeline,
    lookback_days: int,
) -> list[PatternEvidence]:
    """
    For each symptom occurrence, find the closest preceding trigger event
    and build a PatternEvidence record.
    """
    evidence: list[PatternEvidence] = []

    for sym_ev in symptom_events:
        prior = timeline.get_lookback_window(sym_ev, days=lookback_days)
        trig_norm = trigger.lower().strip()

        # Find the trigger event closest to symptom (latest prior trigger)
        trigger_event: HealthEvent | None = None
        for p in reversed(prior):  # reversed = most recent first
            if any(trig_norm in t.lower() for t in p.triggers + p.behaviors):
                trigger_event = p
                break

        if trigger_event is None:
            continue

        lag = timeline.compute_lag_days(trigger_event, sym_ev)
        if lag < 0:
            continue  # Temporal ordering violated — skip

        # Co-occurring variables: all OTHER triggers in the lookback window
        co_vars: list[str] = []
        for p in prior:
            if p.event_id == trigger_event.event_id:
                continue
            co_vars.extend(p.triggers)
            co_vars.extend(p.behaviors)
        co_vars = list(set(v for v in co_vars if trig_norm not in v.lower()))

        ev = PatternEvidence(
            session_id=sym_ev.session_id,
            event_id=sym_ev.event_id,
            occurred_at=sym_ev.timestamp,
            trigger_present=True,
            trigger_observed_at=trigger_event.timestamp,
            lag_days=round(lag, 1),
            symptom_severity=sym_ev.severity.value,
            co_occurring_variables=co_vars[:10],
            raw_excerpt=sym_ev.raw_text[:200],
        )
        evidence.append(ev)

    return evidence


def build_candidates(
    timeline: EventTimeline,
    matrix: CoOccurrenceMatrix,
    min_evidence: int | None = None,
) -> list[CandidatePattern]:
    """
    Main entry point.

    Algorithm:
    1. Group timeline events by symptom.
    2. For each symptom with ≥ min_evidence occurrences:
       a. Use matrix to find top co-occurring triggers.
       b. Use timeline.find_consistent_prior_triggers() to get universal triggers.
       c. Check lag registry for each (symptom, trigger) pair.
       d. Build PatternEvidence for each supporting occurrence.
       e. Construct a CandidatePattern.
    3. De-duplicate: same (symptom, trigger) pair → single candidate.
    4. Sort by occurrence_count desc.
    """
    min_ev = min_evidence or settings.min_evidence_count
    candidates: list[CandidatePattern] = []
    seen: set[tuple[str, str]] = set()

    symptom_groups = timeline.group_by_symptom()

    for symptom, sym_events in symptom_groups.items():
        if len(sym_events) < min_ev:
            continue

        logger.info(
            "hypothesis_builder.symptom_candidate",
            user_id=timeline.user_id,
            symptom=symptom,
            occurrences=len(sym_events),
        )

        # Max lookback for this symptom (registry-aware)
        max_lookback = lag_registry.get_max_lookback(symptom)

        # Find triggers consistent across ALL occurrences
        consistent_triggers = timeline.find_consistent_prior_triggers(
            symptom_events=sym_events,
            lookback_days=max_lookback,
        )

        # Also check matrix for additional high-frequency pairs
        matrix_triggers = matrix.get_triggers_for_symptom(symptom, min_count=min_ev)
        matrix_trigger_names = {cell.trigger for cell in matrix_triggers}

        all_trigger_names = set(consistent_triggers.keys()) | matrix_trigger_names

        for trigger in all_trigger_names:
            pair_key = (symptom, trigger)
            if pair_key in seen:
                continue

            # Consistency score: 1.0 if from consistent_triggers, partial otherwise
            if trigger in consistent_triggers:
                consistency_score = 1.0
            else:
                # Partial: trigger from matrix but not universally consistent
                cell = matrix.get_cell(symptom, trigger)
                consistency_score = (cell.count / len(sym_events)) if cell else 0.5

            if consistency_score < 0.5:
                continue

            # Build evidence records
            evidence = _build_evidence(sym_events, trigger, timeline, max_lookback)

            if len(evidence) < min_ev:
                continue

            # Check lag registry
            mean_lag = (
                sum(e.lag_days for e in evidence) / len(evidence) if evidence else None
            )
            lag_match: LagRegistryMatch | None = lag_registry.lookup(
                symptom=symptom,
                trigger=trigger,
                observed_lag_days=mean_lag,
            )

            # Run variable isolation on confounders
            isolation = _isolator.isolate(evidence, candidate_trigger=trigger)

            candidate = CandidatePattern(
                user_id=timeline.user_id,
                symptom=symptom,
                trigger=trigger,
                evidence=evidence,
                consistency_score=round(consistency_score, 3),
                lag_registry_match=lag_match,
                raw_analysis_notes=isolation.reasoning,
            )
            candidates.append(candidate)
            seen.add(pair_key)

            logger.info(
                "hypothesis_builder.candidate_created",
                user_id=timeline.user_id,
                symptom=symptom,
                trigger=trigger,
                evidence_count=len(evidence),
                consistency=consistency_score,
                registry_match=lag_match.mechanism_name if lag_match else None,
            )

    # Sort: registry matches first, then by occurrence count
    candidates.sort(
        key=lambda c: (
            c.lag_registry_match is not None,
            c.occurrence_count,
            c.consistency_score,
        ),
        reverse=True,
    )

    logger.info(
        "hypothesis_builder.done",
        user_id=timeline.user_id,
        total_candidates=len(candidates),
    )
    return candidates