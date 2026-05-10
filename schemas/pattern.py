"""
schemas/pattern.py — TemporalPattern: the core reasoning output.

A TemporalPattern represents a detected causal relationship between a
trigger and a symptom, with a temporal lag, backed by evidence across
multiple sessions. It has a lifecycle: watching → emerging → confirmed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class PatternStatus(str, Enum):
    WATCHING = "watching"       # N=2, internally tracked
    EMERGING = "emerging"       # N=3 or lag_registry match at N=2
    CONFIRMED = "confirmed"     # N≥4 or skeptic=VERY_HIGH
    REJECTED = "rejected"       # Skeptic hard-gate failed


class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    REJECTED = "rejected"

    def as_score(self) -> float:
        return {
            "very_high": 1.0,
            "high": 0.75,
            "moderate": 0.50,
            "low": 0.25,
            "rejected": 0.0,
        }[self.value]


class PatternEvidence(BaseModel):
    """A single session that supports a pattern."""

    session_id: str
    event_id: UUID
    occurred_at: datetime
    trigger_present: bool
    trigger_observed_at: datetime | None = None
    lag_days: float = Field(
        ..., description="Days from trigger to symptom manifestation"
    )
    symptom_severity: str = ""
    co_occurring_variables: list[str] = Field(
        default_factory=list,
        description="Other factors present — potential confounders"
    )
    raw_excerpt: str = Field("", description="Relevant user quote (≤30 words)")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}


class LagRegistryMatch(BaseModel):
    """A match against the known biological lag registry."""

    mechanism_name: str
    description: str
    lag_min_days: int
    lag_max_days: int
    match_quality: float = Field(..., ge=0.0, le=1.0)
    source: str = "internal_registry"


class SkepticChecks(BaseModel):
    """Result of each rubric check the Skeptic ran."""

    min_evidence_passed: bool = False
    temporal_ordering_passed: bool = False
    trigger_fully_consistent: bool = False
    confounders_present: bool = False
    alternative_equally_plausible: bool = False
    lag_plausible: bool = False
    high_base_rate: bool = False
    dose_response_found: bool = False
    removal_test_passed: bool = False


class SkepticVerdict(BaseModel):
    """Full output of the Skeptic Agent for a single candidate."""

    pattern_id: UUID
    decision: PatternStatus
    confidence: ConfidenceLevel
    checks: SkepticChecks
    positive_signals: list[str] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    dissent_note: str | None = None
    reasoning_trace: str = ""

    model_config = {"json_encoders": {UUID: str}}


class TemporalPattern(BaseModel):
    """
    A confirmed (or in-progress) causal temporal pattern for a user.
    Created by PatternAgent, validated by SkepticAgent, stored persistently.
    """

    pattern_id: UUID = Field(default_factory=uuid4)
    user_id: str

    # ── Identity ──────────────────────────────────────────────────────────
    title: str = Field(..., description="Short human-readable name")
    symptom: str = Field(..., description="The outcome being explained")
    trigger: str = Field(..., description="The identified causal trigger")
    trigger_category: str = Field(
        "", description="food | behavior | stress | environmental"
    )

    # ── Temporal window ───────────────────────────────────────────────────
    lag_days_min: int = Field(ge=0)
    lag_days_max: int = Field(ge=0)
    lag_days_observed: list[float] = Field(
        default_factory=list,
        description="Actual lag days from each evidence session"
    )

    @property
    def lag_days_mean(self) -> float | None:
        if not self.lag_days_observed:
            return None
        return sum(self.lag_days_observed) / len(self.lag_days_observed)

    # ── Evidence ──────────────────────────────────────────────────────────
    evidence: list[PatternEvidence] = Field(default_factory=list)
    occurrence_count: int = Field(0)

    @model_validator(mode="after")
    def sync_occurrence_count(self) -> "TemporalPattern":
        self.occurrence_count = len(self.evidence)
        return self

    # ── Lifecycle ─────────────────────────────────────────────────────────
    status: PatternStatus = Field(PatternStatus.WATCHING)
    confidence: ConfidenceLevel = Field(ConfidenceLevel.LOW)
    skeptic_verdict: SkepticVerdict | None = None

    # ── Registry ──────────────────────────────────────────────────────────
    lag_registry_match: LagRegistryMatch | None = None
    confounders: list[str] = Field(default_factory=list)

    # ── Timestamps ────────────────────────────────────────────────────────
    first_detected_at: datetime = Field(default_factory=datetime.utcnow)
    last_confirmed_at: datetime = Field(default_factory=datetime.utcnow)
    last_reviewed_at: datetime = Field(default_factory=datetime.utcnow)

    def add_evidence(self, ev: PatternEvidence) -> None:
        """Add a new supporting session and update occurrence count."""
        if not any(e.session_id == ev.session_id for e in self.evidence):
            self.evidence.append(ev)
            self.occurrence_count = len(self.evidence)
            if ev.lag_days is not None:
                self.lag_days_observed.append(ev.lag_days)
            self.last_confirmed_at = datetime.utcnow()

    def promote_status(self, min_confirmed: int = 3) -> None:
        """Advance lifecycle based on evidence count and skeptic verdict."""
        if self.status == PatternStatus.REJECTED:
            return
        if self.confidence == ConfidenceLevel.VERY_HIGH:
            self.status = PatternStatus.CONFIRMED
        elif self.occurrence_count >= min_confirmed:
            self.status = PatternStatus.CONFIRMED
        elif self.occurrence_count >= 2 and self.lag_registry_match:
            self.status = PatternStatus.EMERGING
        elif self.occurrence_count >= 2:
            self.status = PatternStatus.EMERGING

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}


class CandidatePattern(BaseModel):
    """
    Lightweight struct emitted by PatternAgent before Skeptic review.
    Not persisted — ephemeral within a single pipeline run.
    """

    candidate_id: UUID = Field(default_factory=uuid4)
    user_id: str
    symptom: str
    trigger: str
    evidence: list[PatternEvidence]
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    lag_registry_match: LagRegistryMatch | None = None
    raw_analysis_notes: str = ""

    @property
    def occurrence_count(self) -> int:
        return len(self.evidence)

    model_config = {"json_encoders": {UUID: str}}