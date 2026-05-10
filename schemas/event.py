"""
schemas/event.py — HealthEvent: the atomic unit of all temporal analysis.

Every user message that contains a health signal is parsed into one or more
HealthEvents. These are the input to the temporal reasoning engine.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class EventType(str, Enum):
    SYMPTOM = "symptom"
    TRIGGER = "trigger"
    BEHAVIOR = "behavior"
    IMPROVEMENT = "improvement"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class BodyLocation(str, Enum):
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    STOMACH = "stomach"
    ABDOMEN = "abdomen"
    LOWER_BACK = "lower_back"
    SKIN_FACE = "skin_face"
    SKIN_BODY = "skin_body"
    SCALP_HAIR = "scalp_hair"
    JOINTS = "joints"
    FULL_BODY = "full_body"
    OTHER = "other"
    UNKNOWN = "unknown"


class TemporalReference(BaseModel):
    """Extracted temporal anchor from user text."""

    explicit_date: datetime | None = None
    relative_phrase: str | None = None  # "last week", "this morning", "again"
    is_ongoing: bool = False
    is_recurrence: bool = False  # signal word: "again", "same as before"

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class HealthEvent(BaseModel):
    """
    Atomic health event extracted from a user message.
    Produced by IntakeAgent, consumed by temporal engine and graph layer.
    """

    event_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)

    # ── Timing ────────────────────────────────────────────────────────────
    # timestamp = when the event OCCURRED (may differ from reported_at)
    timestamp: datetime = Field(...)
    reported_at: datetime = Field(default_factory=datetime.utcnow)
    temporal_ref: TemporalReference = Field(default_factory=TemporalReference)

    # ── Classification ────────────────────────────────────────────────────
    event_type: EventType = Field(EventType.UNKNOWN)
    symptoms: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    behaviors: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    # ── Characterisation ──────────────────────────────────────────────────
    body_location: BodyLocation = Field(BodyLocation.UNKNOWN)
    severity: Severity = Field(Severity.UNKNOWN)
    severity_score: float | None = Field(
        None, ge=0.0, le=10.0,
        description="Numeric severity if extractable (0-10)"
    )
    tags: list[str] = Field(default_factory=list)

    # ── Source ────────────────────────────────────────────────────────────
    raw_text: str = Field(..., min_length=1)
    extraction_confidence: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Confidence that extraction is accurate"
    )

    # ── Graph metadata ────────────────────────────────────────────────────
    linked_pattern_ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def classify_event_type(self) -> "HealthEvent":
        """Auto-set event_type if still UNKNOWN and signals are present."""
        if self.event_type == EventType.UNKNOWN:
            if self.symptoms:
                self.event_type = EventType.SYMPTOM
            elif self.triggers or self.behaviors:
                self.event_type = EventType.TRIGGER
            elif self.improvements:
                self.event_type = EventType.IMPROVEMENT
        return self

    @property
    def timestamp_epoch(self) -> float:
        return self.timestamp.timestamp()

    @property
    def all_signals(self) -> list[str]:
        """Flat list of all health signals for embedding."""
        return self.symptoms + self.triggers + self.behaviors + self.improvements

    def to_embedding_text(self) -> str:
        """Human-readable text suitable for embedding and semantic search."""
        parts = [f"User {self.user_id} reported on {self.timestamp.date()}:"]
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms)}")
        if self.triggers:
            parts.append(f"Triggers: {', '.join(self.triggers)}")
        if self.behaviors:
            parts.append(f"Behaviors: {', '.join(self.behaviors)}")
        if self.body_location != BodyLocation.UNKNOWN:
            parts.append(f"Location: {self.body_location.value}")
        if self.severity != Severity.UNKNOWN:
            parts.append(f"Severity: {self.severity.value}")
        parts.append(f"Raw: {self.raw_text}")
        return " | ".join(parts)

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Flat dict for ChromaDB metadata (no nested objects)."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp_epoch": self.timestamp_epoch,
            "year": self.timestamp.year,
            "month": self.timestamp.month,
            "week_of_year": self.timestamp.isocalendar()[1],
            "day_of_week": self.timestamp.weekday(),
            "hour_of_day": self.timestamp.hour,
            "event_type": self.event_type.value,
            "symptoms": ",".join(self.symptoms),
            "triggers": ",".join(self.triggers),
            "severity": self.severity.value,
            "body_location": self.body_location.value,
            "is_recurrence": str(self.temporal_ref.is_recurrence),
        }

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}


class ParsedSession(BaseModel):
    """Output of IntakeAgent — one session → N HealthEvents."""

    session_id: str
    user_id: str
    raw_messages: list[str]
    events: list[HealthEvent]
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    session_summary: str = ""
    parsed_at: datetime = Field(default_factory=datetime.utcnow)