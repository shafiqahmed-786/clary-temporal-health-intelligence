"""
schemas/session.py — Conversation session schemas and orchestrator state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from schemas.event import HealthEvent, ParsedSession
from schemas.pattern import CandidatePattern, SkepticVerdict, TemporalPattern


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class TriageLevel(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    ESCALATE = "escalate"   # Recommend physician
    EMERGENCY = "emergency"  # Urgent care


class TriageDecision(BaseModel):
    level: TriageLevel = TriageLevel.NORMAL
    reason: str = ""
    escalation_trigger: str | None = None
    recommended_action: str = ""


class ResponseTone(str, Enum):
    EMPATHETIC = "empathetic"
    INFORMATIVE = "informative"
    PATTERN_ALERT = "pattern_alert"
    ESCALATION = "escalation"
    CLARIFYING = "clarifying"


class ClaryResponse(BaseModel):
    """Final structured output delivered to the user."""

    response_id: UUID = Field(default_factory=uuid4)
    session_id: str
    user_id: str
    text: str = Field(..., description="User-facing message text")
    tone: ResponseTone
    patterns_cited: list[UUID] = Field(
        default_factory=list, description="Pattern IDs referenced"
    )
    sessions_cited: list[str] = Field(
        default_factory=list, description="Session IDs used as evidence"
    )
    action_items: list[str] = Field(default_factory=list)
    triage: TriageDecision = Field(default_factory=TriageDecision)
    escalate: bool = False
    disclaimer: str = Field(
        "Clary provides health observations, not medical diagnoses. "
        "Consult a qualified healthcare professional for medical advice."
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}


class OrchestratorStateEnum(str, Enum):
    IDLE = "IDLE"
    INTAKE = "INTAKE"
    CLARIFY = "CLARIFY"
    RETRIEVE = "RETRIEVE"
    PATTERN_DETECT = "PATTERN_DETECT"
    SKEPTIC_REVIEW = "SKEPTIC_REVIEW"
    TRIAGE = "TRIAGE"
    NARRATE = "NARRATE"
    STORE = "STORE"
    ERROR = "ERROR"


class PipelineContext(BaseModel):
    """
    Mutable context object threaded through the entire agent pipeline.
    Each agent reads from and writes to this object.
    """

    # Identity
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    run_id: UUID = Field(default_factory=uuid4)

    # State machine
    state: OrchestratorStateEnum = OrchestratorStateEnum.IDLE
    previous_state: OrchestratorStateEnum | None = None

    # Input
    raw_message: str = ""
    conversation_history: list[Message] = Field(default_factory=list)

    # Intake output
    parsed_session: ParsedSession | None = None
    current_events: list[HealthEvent] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)

    # Retrieval output
    retrieved_sessions: list[dict] = Field(default_factory=list)
    context_window_text: str = ""

    # Pattern detection output
    candidate_patterns: list[CandidatePattern] = Field(default_factory=list)

    # Skeptic output
    skeptic_verdicts: list[SkepticVerdict] = Field(default_factory=list)
    validated_patterns: list[TemporalPattern] = Field(default_factory=list)

    # Triage output
    triage_decision: TriageDecision = Field(default_factory=TriageDecision)

    # Final output
    clary_response: ClaryResponse | None = None

    # Observability
    agent_traces: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    total_tokens_used: int = 0

    def transition(self, new_state: OrchestratorStateEnum) -> None:
        self.previous_state = self.state
        self.state = new_state

    def add_trace(self, agent: str, data: dict) -> None:
        self.agent_traces.append(
            {"agent": agent, "state": self.state.value, "data": data,
             "ts": datetime.utcnow().isoformat()}
        )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}