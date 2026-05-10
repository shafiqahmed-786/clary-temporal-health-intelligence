"""
agents/orchestrator.py — Orchestrator

The central coordinator for the Clary agent pipeline.
Implements the state machine defined in the architecture:

  IDLE → INTAKE → CLARIFY? → RETRIEVE → PATTERN_DETECT →
  SKEPTIC_REVIEW → TRIAGE → NARRATE → STORE → IDLE

Responsibilities:
  - Instantiate and wire all agents
  - Manage PipelineContext lifecycle
  - Persist context to WorkingMemory between turns
  - Coordinate with EpisodicStore and EventGraph post-response
  - Handle errors with graceful degradation (never surface raw errors to user)
  - Emit structured traces for observability (Langfuse / logs)

Multi-turn support:
  - A session with pending clarification is persisted in WorkingMemory.
  - When user responds, context is loaded and pipeline resumes from RETRIEVE.
  - Working memory TTL = 4 hours (configurable).

Dependency injection:
  All dependencies (agents, stores) are injected at construction time.
  This makes the Orchestrator fully testable without live infrastructure.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from agents.base_agent import AgentError
from agents.narrator_agent import NarratorAgent
from agents.pattern_agent import PatternDetectionAgent
from agents.skeptic_agent import SkepticAgent
from agents.triage_agent import TriageAgent
from config import get_settings
from memory.context_builder import ContextBuilder
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.summary_compressor import SummaryCompressor
from memory.working_memory import WorkingMemory
from schemas.event import HealthEvent, EventType, Severity
from schemas.pattern import PatternStatus, TemporalPattern
from schemas.session import (
    ClaryResponse,
    Message,
    OrchestratorStateEnum,
    PipelineContext,
    ResponseTone,
    Role,
    TriageDecision,
    TriageLevel,
)
from schemas.user import UserProfile

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Intake parsing prompt ──────────────────────────────────────────────────────

_INTAKE_SYSTEM_PROMPT = """
You are the Intake Agent for Clary, a health AI assistant.
Parse the user's message and extract structured health information.

Rules:
- Extract ONLY what is explicitly stated or strongly implied.
- Infer timestamps from relative phrases ("this morning", "last week").
- Flag is_recurrence=true if the user says "again", "same", "back again", etc.
- Ask AT MOST 1 clarifying question. Make it the most impactful one.
- Do NOT ask clarifying questions if the message is already clear.
- Set needs_clarification=true ONLY if a key piece of information is genuinely missing
  (e.g., severity is unknown, timing is completely unclear).

Return ONLY valid JSON:
{
  "symptoms": ["list", "of", "symptom", "keywords"],
  "triggers": ["list", "of", "identified", "triggers"],
  "behaviors": ["other relevant behaviors"],
  "severity": "none|mild|moderate|severe|unknown",
  "body_location": "head|stomach|skin_face|scalp_hair|chest|abdomen|lower_back|other|unknown",
  "is_recurrence": true,
  "event_date_iso": "YYYY-MM-DD or null if not determinable",
  "needs_clarification": false,
  "clarification_question": "string or null",
  "session_summary": "1-sentence summary of this session"
}
"""


class Orchestrator:
    """
    Central pipeline coordinator.

    Usage:
        orch = Orchestrator.create()
        response = await orch.process_message(user_id="USR001", message="stomach pain again")
    """

    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        working_memory: WorkingMemory,
        compressor: SummaryCompressor,
        context_builder: ContextBuilder,
        pattern_agent: PatternDetectionAgent,
        skeptic_agent: SkepticAgent,
        triage_agent: TriageAgent,
        narrator_agent: NarratorAgent,
        event_graph: Any | None = None,  # graph.EventGraph — optional
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._wm = working_memory
        self._compressor = compressor
        self._ctx_builder = context_builder
        self._pattern_agent = pattern_agent
        self._skeptic_agent = skeptic_agent
        self._triage_agent = triage_agent
        self._narrator_agent = narrator_agent
        self._event_graph = event_graph
        self._log = structlog.get_logger("orchestrator")

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, event_graph: Any | None = None) -> "Orchestrator":
        """
        Construct the Orchestrator with all dependencies wired.
        Call this once at application startup.
        """
        episodic = EpisodicStore()
        semantic = SemanticStore()
        wm = WorkingMemory()
        compressor = SummaryCompressor(episodic=episodic, working_memory=wm)
        ctx_builder = ContextBuilder(
            episodic=episodic, semantic=semantic, compressor=compressor
        )
        return cls(
            episodic=episodic,
            semantic=semantic,
            working_memory=wm,
            compressor=compressor,
            context_builder=ctx_builder,
            pattern_agent=PatternDetectionAgent(episodic_store=episodic),
            skeptic_agent=SkepticAgent(),
            triage_agent=TriageAgent(),
            narrator_agent=NarratorAgent(),
            event_graph=event_graph,
        )

    # ── Main entry point ───────────────────────────────────────────────────

    async def process_message(
        self,
        user_id: str,
        message: str,
        session_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> ClaryResponse:
        """
        Process a single user message through the full pipeline.

        Returns a ClaryResponse (always — errors produce safe fallbacks).
        """
        # Check rate limit
        allowed, remaining = await self._wm.check_rate_limit(user_id)
        if not allowed:
            return self._rate_limit_response(user_id, session_id or str(uuid4()))

        # Load or create PipelineContext
        ctx = await self._load_or_create_context(
            user_id=user_id,
            message=message,
            session_id=session_id,
            profile=profile,
        )

        self._log.info(
            "orchestrator.pipeline_start",
            user_id=user_id,
            session_id=ctx.session_id,
            state=ctx.state.value,
            message_preview=message[:80],
        )

        try:
            ctx = await self._run_pipeline(ctx)
        except Exception as exc:
            self._log.error(
                "orchestrator.pipeline_error",
                error=str(exc),
                user_id=user_id,
                session_id=ctx.session_id,
            )
            ctx.errors.append(str(exc))
            ctx.clary_response = self._error_fallback(ctx)

        # Persist state
        await self._wm.save_context(ctx)

        if ctx.clary_response:
            await self._wm.append_message(
                ctx.session_id,
                Message(role=Role.USER, content=message),
            )
            await self._wm.append_message(
                ctx.session_id,
                Message(role=Role.ASSISTANT, content=ctx.clary_response.text),
            )

        # Post-session async work (persist to episodic + graph)
        if ctx.state == OrchestratorStateEnum.STORE:
            await self._store_session(ctx)

        ctx.completed_at = datetime.utcnow()
        self._log.info(
            "orchestrator.pipeline_done",
            user_id=user_id,
            session_id=ctx.session_id,
            final_state=ctx.state.value,
            total_tokens=ctx.total_tokens_used,
            patterns_validated=len(ctx.validated_patterns),
        )

        return ctx.clary_response or self._error_fallback(ctx)

    # ── State machine ──────────────────────────────────────────────────────

    async def _run_pipeline(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the state machine from current state through to STORE."""

        # ── INTAKE ────────────────────────────────────────────────────────
        ctx.transition(OrchestratorStateEnum.INTAKE)
        ctx = await self._intake(ctx)

        # ── CLARIFY (optional) ────────────────────────────────────────────
        if ctx.needs_clarification:
            ctx.transition(OrchestratorStateEnum.CLARIFY)
            ctx = await self._clarify(ctx)
            # If we just asked a clarifying question, return early
            # The next user message will resume from RETRIEVE
            if ctx.state == OrchestratorStateEnum.CLARIFY:
                return ctx

        # ── RETRIEVE ──────────────────────────────────────────────────────
        ctx.transition(OrchestratorStateEnum.RETRIEVE)
        ctx = await self._retrieve(ctx)

        # ── PATTERN_DETECT ────────────────────────────────────────────────
        ctx.transition(OrchestratorStateEnum.PATTERN_DETECT)
        ctx = await self._pattern_agent.process(ctx)

        # ── SKEPTIC_REVIEW ────────────────────────────────────────────────
        if ctx.candidate_patterns:
            ctx.transition(OrchestratorStateEnum.SKEPTIC_REVIEW)
            ctx = await self._skeptic_agent.process(ctx)

        # ── TRIAGE ────────────────────────────────────────────────────────
        ctx.transition(OrchestratorStateEnum.TRIAGE)
        ctx = await self._triage_agent.process(ctx)

        # ── NARRATE ───────────────────────────────────────────────────────
        ctx.transition(OrchestratorStateEnum.NARRATE)
        ctx = await self._narrator_agent.process(ctx)

        # ── STORE ─────────────────────────────────────────────────────────
        ctx.transition(OrchestratorStateEnum.STORE)

        return ctx

    # ── INTAKE implementation ──────────────────────────────────────────────

    async def _intake(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse the raw message into structured HealthEvent(s) using LLM.
        Also detects recurrence signals ("again", "same as before").
        """
        # Check if this is the answer to a clarification question
        pending_clarification = await self._wm.get_clarification_pending(ctx.session_id)
        if pending_clarification:
            # User answered our clarifying question — resume from intake
            await self._wm.clear_clarification_pending(ctx.session_id)
            ctx.needs_clarification = False
            self._log.info(
                "orchestrator.clarification_answered",
                session_id=ctx.session_id,
            )

        # Use the NarratorAgent's LLM infrastructure for intake (it has retry etc.)
        # We call the LLM directly here via a lightweight parse call
        import openai as _openai
        import time

        oai = _openai.AsyncOpenAI(api_key=settings.openai_api_key)
        t0 = time.monotonic()

        try:
            response = await oai.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": _INTAKE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Today's date: {datetime.utcnow().strftime('%Y-%m-%d')}\n"
                            f"User message: {ctx.raw_message}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            ctx.total_tokens_used += (response.usage.total_tokens if response.usage else 0)
        except Exception as exc:
            self._log.error("orchestrator.intake_failed", error=str(exc))
            ctx.current_events = []
            ctx.needs_clarification = False
            return ctx

        latency_ms = (time.monotonic() - t0) * 1000
        self._log.debug(
            "orchestrator.intake_llm", latency_ms=round(latency_ms)
        )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        # Build HealthEvent from parsed intake
        event_date_str = parsed.get("event_date_iso")
        try:
            event_ts = (
                datetime.fromisoformat(event_date_str)
                if event_date_str
                else datetime.utcnow()
            )
        except (ValueError, TypeError):
            event_ts = datetime.utcnow()

        sev_str = parsed.get("severity", "unknown")
        try:
            severity = Severity(sev_str)
        except ValueError:
            severity = Severity.UNKNOWN

        event = HealthEvent(
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            timestamp=event_ts,
            symptoms=parsed.get("symptoms", []),
            triggers=parsed.get("triggers", []),
            behaviors=parsed.get("behaviors", []),
            severity=severity,
            raw_text=ctx.raw_message,
        )

        ctx.current_events = [event]
        ctx.needs_clarification = bool(parsed.get("needs_clarification", False))

        clarification_q = parsed.get("clarification_question")
        if ctx.needs_clarification and clarification_q:
            ctx.clarification_questions = [clarification_q]

        ctx.add_trace(
            agent="intake",
            data={
                "symptoms": event.symptoms,
                "triggers": event.triggers,
                "severity": event.severity.value,
                "is_recurrence": event.temporal_ref.is_recurrence,
                "needs_clarification": ctx.needs_clarification,
            },
        )
        return ctx

    # ── CLARIFY implementation ─────────────────────────────────────────────

    async def _clarify(self, ctx: PipelineContext) -> PipelineContext:
        """
        Send clarifying question to user and persist pending state.
        The response will arrive as the next user message.
        """
        question = (
            ctx.clarification_questions[0]
            if ctx.clarification_questions
            else "Could you give me a bit more detail about what you're experiencing?"
        )

        ctx.clary_response = ClaryResponse(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            text=question,
            tone=ResponseTone.CLARIFYING,
            triage=TriageDecision(),
            escalate=False,
        )

        await self._wm.set_clarification_pending(
            ctx.session_id, ctx.clarification_questions
        )

        self._log.info(
            "orchestrator.clarification_sent",
            session_id=ctx.session_id,
            question=question[:80],
        )
        # State stays CLARIFY — pipeline exits here
        return ctx

    # ── RETRIEVE implementation ────────────────────────────────────────────

    async def _retrieve(self, ctx: PipelineContext) -> PipelineContext:
        """
        Assemble context window from all memory tiers.
        Also retrieves confirmed patterns from cache or store.
        """
        # Get user profile — in production loaded from PostgreSQL
        # For now, create a minimal profile from session data
        profile = UserProfile(user_id=ctx.user_id)

        # Build full context window
        ctx.context_window_text = await self._ctx_builder.build_pattern_analysis_context(
            user_id=ctx.user_id,
            current_message=ctx.raw_message,
            profile=profile,
        )

        # Retrieve confirmed patterns from cache
        cached_patterns_raw = await self._wm.get_cached_patterns(ctx.user_id)
        if cached_patterns_raw:
            try:
                ctx.validated_patterns = [
                    TemporalPattern.model_validate(p) for p in cached_patterns_raw
                ]
                self._log.debug(
                    "orchestrator.patterns_from_cache",
                    user_id=ctx.user_id,
                    count=len(ctx.validated_patterns),
                )
            except Exception:
                ctx.validated_patterns = []

        ctx.add_trace(
            agent="retrieve",
            data={
                "context_chars": len(ctx.context_window_text),
                "cached_patterns": len(ctx.validated_patterns),
            },
        )
        return ctx

    # ── STORE implementation ───────────────────────────────────────────────

    async def _store_session(self, ctx: PipelineContext) -> None:
        """
        Persist all session outputs asynchronously.
        Called after response is sent — does not block the user.
        """
        try:
            # Store events to episodic memory
            if ctx.current_events:
                await self._episodic.store_events(ctx.current_events)
                self._log.info(
                    "orchestrator.stored_events",
                    session_id=ctx.session_id,
                    count=len(ctx.current_events),
                )

            # Update confirmed pattern cache
            if ctx.validated_patterns:
                confirmed = [
                    p for p in ctx.validated_patterns
                    if p.status in (PatternStatus.CONFIRMED, PatternStatus.EMERGING)
                ]
                if confirmed:
                    await self._wm.cache_patterns(
                        ctx.user_id,
                        [p.model_dump(mode="json") for p in confirmed],
                    )

            # Invalidate summary cache if new events added
            if ctx.current_events:
                await self._compressor.invalidate(ctx.user_id)

            # Update event graph (if available)
            if self._event_graph and ctx.current_events:
                try:
                    await self._event_graph.ingest_pipeline_context(ctx)
                except Exception as graph_exc:
                    # Graph failure never blocks pipeline
                    self._log.warning(
                        "orchestrator.graph_update_failed",
                        error=str(graph_exc),
                        session_id=ctx.session_id,
                    )

        except Exception as exc:
            self._log.error(
                "orchestrator.store_failed",
                error=str(exc),
                session_id=ctx.session_id,
            )

    # ── Context management ─────────────────────────────────────────────────

    async def _load_or_create_context(
        self,
        user_id: str,
        message: str,
        session_id: str | None,
        profile: UserProfile | None,
    ) -> PipelineContext:
        """
        Load existing PipelineContext from WorkingMemory (multi-turn),
        or create a new one for a fresh session.
        """
        if session_id:
            existing = await self._wm.load_context(session_id)
            if existing:
                # Resume existing session — update raw message
                existing.raw_message = message
                existing.conversation_history.append(
                    Message(role=Role.USER, content=message)
                )
                # Reset per-turn outputs
                existing.candidate_patterns = []
                existing.skeptic_verdicts = []
                existing.clary_response = None
                return existing

        return PipelineContext(
            session_id=session_id or str(uuid4()),
            user_id=user_id,
            raw_message=message,
            conversation_history=[Message(role=Role.USER, content=message)],
        )

    # ── Fallback responses ─────────────────────────────────────────────────

    @staticmethod
    def _error_fallback(ctx: PipelineContext) -> ClaryResponse:
        return ClaryResponse(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            text=(
                "I'm here and I noted what you shared. "
                "I ran into a small technical issue processing this — "
                "please try again in a moment."
            ),
            tone=ResponseTone.EMPATHETIC,
            triage=ctx.triage_decision,
            escalate=False,
        )

    @staticmethod
    def _rate_limit_response(user_id: str, session_id: str) -> ClaryResponse:
        return ClaryResponse(
            session_id=session_id,
            user_id=user_id,
            text=(
                "You've sent quite a few messages recently. "
                "Please wait a bit before sending another."
            ),
            tone=ResponseTone.INFORMATIVE,
            triage=TriageDecision(),
            escalate=False,
        )