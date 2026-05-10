"""
agents/base_agent.py — BaseAgent

Abstract base class for all Clary agents.
Provides:
  - Async LLM call with JSON structured output
  - Tenacity retry with exponential backoff
  - Automatic AgentTrace population
  - Token usage tracking
  - Structured logging throughout

Every concrete agent:
  1. Extends BaseAgent
  2. Defines AGENT_NAME and optional SYSTEM_PROMPT
  3. Implements process(ctx) → PipelineContext
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar
from uuid import UUID

import openai
import structlog
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings
from schemas.agent import AgentTrace, LLMUsage
from schemas.session import PipelineContext

logger = structlog.get_logger(__name__)
settings = get_settings()

T = TypeVar("T", bound=BaseModel)


class AgentError(Exception):
    """Raised when an agent fails after all retries."""
    pass


class BaseAgent(ABC):
    """
    Abstract base for all Clary agents.

    Subclasses must implement:
      - AGENT_NAME: str
      - process(ctx: PipelineContext) → PipelineContext
      - (optionally) SYSTEM_PROMPT: str
    """

    AGENT_NAME: str = "base_agent"
    SYSTEM_PROMPT: str = "You are a helpful health AI assistant."

    def __init__(self) -> None:
        self._openai: openai.AsyncOpenAI | None = None
        self._log = structlog.get_logger(self.__class__.__name__)

    def _get_openai(self) -> openai.AsyncOpenAI:
        if self._openai is None:
            self._openai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Run this agent's logic, mutate ctx, and return it.
        All agents receive and return the shared PipelineContext.
        """
        ...

    # ── Core LLM call ──────────────────────────────────────────────────────

    async def _llm_call(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        run_id: UUID,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, AgentTrace]:
        """
        Make an async LLM call with retry and full trace capture.

        Args:
            messages:        OpenAI chat messages list
            session_id:      For trace attribution
            run_id:          Pipeline run identifier
            response_schema: Optional JSON schema hint (injected into system prompt)
            temperature:     Override global default
            max_tokens:      Override global default

        Returns:
            (raw_content_string, AgentTrace)
        """
        trace = AgentTrace(
            agent_name=self.AGENT_NAME,
            model_version=settings.openai_model,
            session_id=session_id,
            run_id=run_id,
        )

        temp = temperature if temperature is not None else settings.openai_temperature
        tokens = max_tokens or settings.openai_max_tokens

        # Inject JSON schema hint into last system message if provided
        if response_schema:
            schema_hint = (
                "\n\nYou MUST respond with a single valid JSON object. "
                "Do NOT include markdown fences, preamble, or trailing text. "
                f"Required schema: {json.dumps(response_schema, indent=2)}"
            )
            # Find or create system message
            sys_msgs = [m for m in messages if m["role"] == "system"]
            if sys_msgs:
                sys_msgs[-1]["content"] += schema_hint
            else:
                messages.insert(0, {"role": "system", "content": schema_hint})

        last_error: Exception | None = None
        attempt = 0
        t0 = time.monotonic()

        try:
            async for attempt_ctx in AsyncRetrying(
                stop=stop_after_attempt(settings.max_agent_retries),
                wait=wait_exponential(
                    multiplier=settings.retry_base_delay,
                    max=settings.retry_max_delay,
                ),
                retry=retry_if_exception_type(
                    (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
                ),
                reraise=False,
            ):
                with attempt_ctx:
                    attempt += 1
                    trace.attempt_number = attempt
                    self._log.info(
                        "agent.llm_call.attempt",
                        agent=self.AGENT_NAME,
                        attempt=attempt,
                        session_id=session_id,
                    )

                    client = self._get_openai()
                    response = await client.chat.completions.create(
                        model=settings.openai_model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temp,
                        max_tokens=tokens,
                        response_format={"type": "json_object"},
                    )

                    content = response.choices[0].message.content or ""
                    usage = response.usage

                    trace.raw_llm_response = content
                    trace.usage = LLMUsage(
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        total_tokens=usage.total_tokens if usage else 0,
                    )
                    latency = (time.monotonic() - t0) * 1000
                    trace.mark_complete(latency_ms=latency, succeeded=True)

                    self._log.info(
                        "agent.llm_call.success",
                        agent=self.AGENT_NAME,
                        latency_ms=round(latency),
                        tokens=trace.usage.total_tokens,
                        attempt=attempt,
                    )
                    return content, trace

        except RetryError as exc:
            last_error = exc.__cause__ or exc
        except Exception as exc:
            last_error = exc

        latency = (time.monotonic() - t0) * 1000
        trace.mark_complete(latency_ms=latency, succeeded=False)
        trace.error = str(last_error)
        self._log.error(
            "agent.llm_call.failed",
            agent=self.AGENT_NAME,
            error=str(last_error),
            attempts=attempt,
        )
        raise AgentError(
            f"{self.AGENT_NAME} failed after {attempt} attempts: {last_error}"
        ) from last_error

    # ── JSON parsing helpers ───────────────────────────────────────────────

    def _parse_json(self, raw: str, context: str = "") -> dict[str, Any]:
        """
        Parse LLM JSON response. Strips markdown fences if present.
        Raises AgentError on parse failure.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self._log.error(
                "agent.parse_error",
                agent=self.AGENT_NAME,
                context=context,
                raw_preview=raw[:200],
                error=str(exc),
            )
            raise AgentError(
                f"{self.AGENT_NAME}: JSON parse failed. Raw: {raw[:200]}"
            ) from exc

    def _parse_model(self, raw: str, model_class: type[T]) -> T:
        """Parse raw JSON string into a Pydantic model."""
        data = self._parse_json(raw, context=model_class.__name__)
        return model_class.model_validate(data)

    # ── System prompt builder ──────────────────────────────────────────────

    def _build_messages(
        self,
        user_content: str,
        system_override: str | None = None,
        extra_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Construct the messages list for an LLM call."""
        system = system_override or self.SYSTEM_PROMPT
        if extra_context:
            user_content = f"{extra_context}\n\n---\n\n{user_content}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    # ── Trace helpers ──────────────────────────────────────────────────────

    def _record_trace(self, ctx: PipelineContext, trace: AgentTrace, summary: str) -> None:
        """Log the agent trace to PipelineContext."""
        trace.output_summary = summary
        ctx.add_trace(
            agent=self.AGENT_NAME,
            data={
                "trace_id": str(trace.trace_id),
                "latency_ms": trace.latency_ms,
                "tokens": trace.usage.total_tokens,
                "succeeded": trace.succeeded,
                "summary": summary,
            },
        )
        ctx.total_tokens_used += trace.usage.total_tokens