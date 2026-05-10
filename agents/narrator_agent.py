"""
agents/narrator_agent.py — NarratorAgent

Synthesises all pipeline outputs into a single, human-toned response.

Tone selection matrix:
  EMERGENCY triage            → ESCALATION tone (urgent, directive)
  ESCALATE triage             → ESCALATION tone
  CONFIRMED patterns present  → PATTERN_ALERT tone (assertive, cite dates)
  EMERGING patterns present   → INFORMATIVE tone (observational)
  No patterns, first session  → EMPATHETIC tone (curious, supportive)
  Clarification needed        → CLARIFYING tone

Calibrated language by confidence:
  VERY_HIGH  → "this is clearly a pattern", "consistently causes"
  HIGH       → "this appears to be a pattern", "very likely connected"
  MODERATE   → "there might be a pattern here", "worth watching"
  LOW/WATCH  → "I've noticed this once before", "something to keep an eye on"

Evidence citation rules:
  - Always cite specific session dates (not "previously" or "in the past")
  - Quote the observed lag when relevant ("about 48 hours after")
  - Name the biological mechanism when a registry match exists
  - Never use the word "diagnosis" or claim to diagnose

Disclaimer: always appended at end.
"""

from __future__ import annotations

import structlog

from agents.base_agent import AgentError, BaseAgent
from config import get_settings
from schemas.pattern import ConfidenceLevel, PatternStatus, TemporalPattern
from schemas.session import (
    ClaryResponse,
    PipelineContext,
    ResponseTone,
    TriageLevel,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are Clary, a warm and intelligent personal health companion.
You synthesise health observations from a user's history into
a concise, empathetic, and evidence-based response.

══ CORE RULES ══
1. NEVER diagnose. Frame everything as observation or hypothesis.
2. ALWAYS cite specific dates when referencing past sessions.
3. Language must be CALIBRATED to evidence strength (see below).
4. Keep responses CONCISE: 2–4 sentences for simple observations,
   4–8 sentences max when citing patterns.
5. End with a clear action item when a pattern is present.
6. For ESCALATION: lead with the concern, then give action. Do not bury it.

══ CALIBRATED LANGUAGE ══
confidence=very_high : "this is a clear pattern", "consistently causes", "each time"
confidence=high      : "this appears to be a pattern", "very likely connected"
confidence=moderate  : "there might be a pattern here", "possibly connected"
confidence=low/watch : "something to keep an eye on", "noticed once before"
No patterns yet      : "I don't see a pattern yet, but I'll keep track"

══ TONE GUIDE ══
EMPATHETIC   : First sessions, no patterns. Warm, curious, supportive.
INFORMATIVE  : Emerging patterns. Observational, non-alarming.
PATTERN_ALERT: Confirmed patterns. Assertive, cite dates, name mechanism.
ESCALATION   : Red flags. Direct, concerned, clear action required.
CLARIFYING   : Need more information. One focused question.

══ WHAT TO INCLUDE ══
- Acknowledge the current complaint
- Reference relevant past sessions BY DATE
- State the pattern with calibrated confidence
- Name biological mechanism if known (e.g., "this is likely Telogen Effluvium")
- Give one concrete action item
- If escalation: clearly recommend seeing a doctor

══ WHAT TO EXCLUDE ══
- Do not repeat the disclaimer (it is appended automatically)
- Do not pad with generic health advice unrelated to this user's patterns
- Do not use the word "diagnosis"
- Do not list symptoms back to the user verbatim

Return ONLY a valid JSON object:
{{
  "text": "string — the full response to the user",
  "tone": "empathetic|informative|pattern_alert|escalation|clarifying",
  "action_items": ["list", "of", "specific", "actions"],
  "sessions_cited": ["session_id_1", "session_id_2"],
  "patterns_cited": ["pattern_id_1"]
}}
""".strip()

_USER_TEMPLATE = """
══ USER PROFILE ══
{profile_text}

══ CURRENT MESSAGE ══
"{current_message}"

══ VALIDATED PATTERNS (from Skeptic Agent) ══
{patterns_text}

══ TRIAGE DECISION ══
Level: {triage_level}
Reason: {triage_reason}
Recommended action: {triage_action}

══ CONTEXT WINDOW (recent history) ══
{context_snippet}

Write the response now. Tone guidance: {tone_hint}.
"""


class NarratorAgent(BaseAgent):
    """
    Generates the final user-facing ClaryResponse.

    Inputs:
      - ctx.validated_patterns
      - ctx.triage_decision
      - ctx.context_window_text
      - ctx.raw_message
      - ctx.current_events

    Outputs:
      - ctx.clary_response
    """

    AGENT_NAME = "narrator_agent"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        self._log.info(
            "narrator_agent.start",
            user_id=ctx.user_id,
            validated_patterns=len(ctx.validated_patterns),
            triage_level=ctx.triage_decision.level.value,
        )

        tone_hint = self._select_tone_hint(ctx)
        patterns_text = self._format_patterns(ctx.validated_patterns)
        profile_text = self._format_profile(ctx)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    profile_text=profile_text,
                    current_message=ctx.raw_message[:500],
                    patterns_text=patterns_text,
                    triage_level=ctx.triage_decision.level.value.upper(),
                    triage_reason=ctx.triage_decision.reason,
                    triage_action=ctx.triage_decision.recommended_action,
                    context_snippet=ctx.context_window_text[:2000],
                    tone_hint=tone_hint,
                ),
            },
        ]

        try:
            raw, trace = await self._llm_call(
                messages=messages,
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                temperature=0.3,  # Slightly higher for natural language
            )
        except AgentError as exc:
            self._log.error("narrator_agent.llm_failed", error=str(exc))
            ctx.clary_response = self._fallback_response(ctx)
            return ctx

        self._record_trace(ctx, trace, "Narrator response generated")

        parsed = self._parse_json(raw, context="NarratorAgent")

        tone_str = parsed.get("tone", "informative").lower()
        try:
            tone = ResponseTone(tone_str)
        except ValueError:
            tone = ResponseTone.INFORMATIVE

        # Escalation override: if triage says escalate, force tone
        if ctx.triage_decision.level in (TriageLevel.ESCALATE, TriageLevel.EMERGENCY):
            tone = ResponseTone.ESCALATION

        ctx.clary_response = ClaryResponse(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            text=parsed.get("text", self._fallback_response(ctx).text),
            tone=tone,
            patterns_cited=[
                p.pattern_id
                for p in ctx.validated_patterns
                if str(p.pattern_id) in parsed.get("patterns_cited", [])
            ],
            sessions_cited=parsed.get("sessions_cited", []),
            action_items=parsed.get("action_items", []),
            triage=ctx.triage_decision,
            escalate=ctx.triage_decision.level in (TriageLevel.ESCALATE, TriageLevel.EMERGENCY),
        )

        self._log.info(
            "narrator_agent.done",
            user_id=ctx.user_id,
            tone=tone.value,
            escalate=ctx.clary_response.escalate,
            response_chars=len(ctx.clary_response.text),
        )
        return ctx

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _select_tone_hint(ctx: PipelineContext) -> str:
        """Choose the appropriate tone guidance string for the LLM."""
        if ctx.triage_decision.level in (TriageLevel.ESCALATE, TriageLevel.EMERGENCY):
            return (
                "ESCALATION — lead with the concern, be direct, give a clear action. "
                "Do not soften or bury the escalation."
            )

        confirmed = [
            p for p in ctx.validated_patterns
            if p.status == PatternStatus.CONFIRMED
        ]
        emerging = [
            p for p in ctx.validated_patterns
            if p.status == PatternStatus.EMERGING
        ]

        if confirmed:
            best = max(confirmed, key=lambda p: p.confidence.as_score())
            return (
                f"PATTERN_ALERT — assert the confirmed pattern clearly. "
                f"Cite the specific dates in evidence. "
                f"Confidence: {best.confidence.value}. "
                f"Mechanism: {best.lag_registry_match.mechanism_name if best.lag_registry_match else 'unknown'}."
            )

        if emerging:
            return (
                "INFORMATIVE — note the emerging pattern observationally. "
                "Use hedged language ('appears to be', 'worth watching'). "
                "Do not alarm the user."
            )

        return (
            "EMPATHETIC — acknowledge the symptom warmly. "
            "No patterns yet, so focus on understanding. "
            "Keep it brief and supportive."
        )

    @staticmethod
    def _format_patterns(patterns: list[TemporalPattern]) -> str:
        if not patterns:
            return "No validated patterns at this time."

        lines = []
        for p in patterns:
            evidence_dates = [
                e.occurred_at.strftime("%Y-%m-%d") for e in p.evidence[:5]
            ]
            lag_str = (
                f"{p.lag_days_min}–{p.lag_days_max} days"
                if p.lag_days_max > 0
                else "same day"
            )
            lines.append(
                f"Pattern [{p.status.value.upper()}, confidence={p.confidence.value}]: "
                f"{p.trigger} → {p.symptom} "
                f"(lag: {lag_str}, N={p.occurrence_count})"
            )
            lines.append(f"  Evidence dates: {', '.join(evidence_dates)}")
            if p.lag_registry_match:
                lines.append(
                    f"  Mechanism: {p.lag_registry_match.mechanism_name} — "
                    f"{p.lag_registry_match.description[:150]}..."
                )
            if p.confounders:
                lines.append(f"  Confounders noted: {', '.join(p.confounders[:3])}")
            lines.append(f"  Pattern ID: {p.pattern_id}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_profile(ctx: PipelineContext) -> str:
        return (
            f"User: {ctx.user_id} | "
            f"Session: {ctx.session_id}"
        )

    def _fallback_response(self, ctx: PipelineContext) -> ClaryResponse:
        """Safe fallback if LLM call fails."""
        text = (
            "I heard you. Let me keep track of this so I can spot any patterns over time."
        )
        if ctx.triage_decision.level == TriageLevel.EMERGENCY:
            text = (
                "⚠️ Please seek immediate medical attention. "
                "If you are experiencing a medical emergency, call emergency services now."
            )
        elif ctx.triage_decision.level == TriageLevel.ESCALATE:
            text = (
                f"I want to flag this for you — "
                f"{ctx.triage_decision.recommended_action}"
            )

        return ClaryResponse(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            text=text,
            tone=ResponseTone.EMPATHETIC,
            triage=ctx.triage_decision,
            escalate=ctx.triage_decision.level in (TriageLevel.ESCALATE, TriageLevel.EMERGENCY),
        )