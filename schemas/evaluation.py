"""
schemas/evaluation.py — Evaluation harness schemas.

The 8 golden patterns from the dataset form the test suite.
These schemas capture ground truth and system predictions for scoring.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GoldenPattern(BaseModel):
    """Ground truth entry from hidden_patterns_reference in the dataset."""

    pattern_id: str
    user_id: str
    symptom: str
    trigger: str
    lag_days_min: int
    lag_days_max: int
    mechanism: str
    first_trigger_session: str
    first_symptom_session: str
    difficulty: str  # easy | medium | hard
    notes: str = ""


class PatternMatchResult(BaseModel):
    """Result of comparing a system-detected pattern to a golden pattern."""

    golden_pattern_id: str
    detected_pattern_id: UUID | None = None
    is_true_positive: bool = False

    # Temporal accuracy
    lag_within_tolerance: bool | None = None
    lag_tolerance_days: int = 7
    detected_lag_days: float | None = None
    golden_lag_days_mid: float | None = None

    # Signal accuracy
    symptom_match: bool = False
    trigger_match: bool = False
    trigger_fuzzy_match: bool = False

    # Confidence
    detected_confidence: str | None = None
    detected_status: str | None = None

    match_notes: str = ""

    model_config = {"json_encoders": {UUID: str}}


class EvalRunResult(BaseModel):
    """Aggregate result of one full evaluation run."""

    run_id: UUID = Field(default_factory=uuid4)
    run_at: datetime = Field(default_factory=datetime.utcnow)
    dataset_path: str = ""
    model_version: str = ""

    # Per-pattern results
    pattern_results: list[PatternMatchResult] = Field(default_factory=list)

    # Core metrics
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    # Temporal accuracy
    temporal_accuracy: float | None = None  # % within tolerance

    # Skeptic accuracy
    false_positive_rate: float | None = None

    # Thresholds
    min_f1_threshold: float = 0.85
    passed: bool = False

    def compute_metrics(self) -> None:
        total_golden = len(self.pattern_results)
        if total_golden == 0:
            return

        tp = sum(1 for r in self.pattern_results if r.is_true_positive)
        all_detected = sum(
            1 for r in self.pattern_results if r.detected_pattern_id is not None
        )
        fp = all_detected - tp

        self.precision = tp / all_detected if all_detected > 0 else 0.0
        self.recall = tp / total_golden
        if self.precision and self.recall:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1 = 0.0

        temporal_results = [
            r for r in self.pattern_results
            if r.is_true_positive and r.lag_within_tolerance is not None
        ]
        if temporal_results:
            self.temporal_accuracy = sum(
                1 for r in temporal_results if r.lag_within_tolerance
            ) / len(temporal_results)

        self.false_positive_rate = fp / all_detected if all_detected > 0 else 0.0
        self.passed = (self.f1 or 0.0) >= self.min_f1_threshold

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}