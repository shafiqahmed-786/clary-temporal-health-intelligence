"""
schemas/agent.py — Agent response envelopes and trace structures.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class AgentTrace(BaseModel):
    """Full observability record for a single agent invocation."""

    trace_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    model_version: str
    session_id: str
    run_id: UUID

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    latency_ms: float | None = None

    # I/O
    input_summary: str = ""
    output_summary: str = ""
    raw_llm_response: str = ""
    parsed_output: dict[str, Any] = Field(default_factory=dict)

    # Cost / usage
    usage: LLMUsage = Field(default_factory=LLMUsage)

    # Retries
    attempt_number: int = 1
    max_attempts: int = 3

    # Errors
    error: str | None = None
    succeeded: bool = True

    def mark_complete(self, latency_ms: float, succeeded: bool = True) -> None:
        self.completed_at = datetime.utcnow()
        self.latency_ms = latency_ms
        self.succeeded = succeeded

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat(), UUID: str}}


class AgentResponse(BaseModel):
    """Generic response envelope returned by every agent."""

    agent_name: str
    trace: AgentTrace
    output: dict[str, Any] = Field(default_factory=dict)
    succeeded: bool = True
    error_message: str | None = None

    model_config = {"json_encoders": {UUID: str}}