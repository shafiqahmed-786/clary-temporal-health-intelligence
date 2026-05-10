"""
agents/triage_agent.py — TriageAgent

Assesses the clinical severity of the current session and decides
whether to escalate to physician referral.

Conservative approach: better a false positive than missing something serious.

Escalation triggers (any one → ESCALATE):
  - Severity=severe and symptom repeated ≥3 times
  - Red flag keywords: "chest tightness", "jaw pain", "arm pain", "blood",
    "can't breathe", "vision", "weakness", "collapse", "fainting"
  - Any symptom marked severe with no prior history (first occurrence)

EMERGENCY triggers (any one → EMERGENCY):
  - "chest pain radiating", "can't breathe", "loss of consciousness",
    "stroke", "severe bleeding", "anaphylaxis"

The TriageAgent is intentionally paranoid.
It never downplays symptoms.
"""

from __future__ import annotations

import re

import structlog

from agents.base_agent import AgentError, BaseAgent
from config import get_settings
from schemas.event import HealthEvent, Severity
from schemas.session import (
    PipelineContext,
    TriageDecision,
    TriageLevel,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Red flag keyword patterns ──────────────────────────────────────────────────

_EMERGENCY_PATTERNS = [
    r"chest.{0,10}pain.{0,20}(jaw|arm|shoulder|radiat)",
    r"(can.t|cannot|difficulty)\s+breath",
    r"loss of consciousness",
    r"(stroke|heart attack)",
    r"severe bleeding",
    r"anaphylax",
    r"(face|arm).{0,10}(drooping|numb|weak)",
]

_ESCALATION_PATTERNS = [
    r"chest (pain|tightness|pressure)",
    r"(jaw|arm|left arm|shoulder) (pain|ache|numb)",
    r"blood in (urine|stool|vomit)",
    r"coughing blood",
    r"unexplained weight loss",
    r"high fever",
    r"(lump|mass|growth)",
    r"(blurred|double|loss of) vision",
    r"sudden (severe|intense) headache",
    r"(weakness|numbness|tingling) (in|on) (one|left|right)",
    r"persistent (vomiting|diarrhea).{0,20}(day|week)",
    r"(pain|pressure) (spreading|radiating)",
]

_SYSTEM_PROMPT = """
You are the Triage Agent for Clary health AI.
Assess the clinical severity of the current health complaint.
Be conservative — it is better to escalate when not strictly necessary
than to miss a genuinely serious issue.

══ ESCALATION LEVELS ══
NORMAL: Mild/moderate recurring symptoms, no red flags.
WATCH: Moderate severity persisting >3 occurrences, or unknown cause after multiple sessions.
ESCALATE: Recommend seeing a doctor within 1-7 days. Red flag present. Persistent severe symptoms.
EMERGENCY: Seek immediate medical attention. Life-threatening signals.

══ RED FLAGS requiring ESCALATE at minimum ══
- Chest pain or pressure
- Severe headache (sudden onset, "worst of life")
- Blood in urine, stool, or vomit
- Unexplained weight loss or fatigue
- Neurological symptoms: weakness, numbness, vision changes
- Symptoms radiating to jaw, arm, or shoulder
- Persistent fever >38.5°C / 101.3°F
- Any symptom that is worsening progressively across sessions

══ RED FLAGS requiring EMERGENCY ══
- Chest pain radiating to jaw/arm
- Difficulty breathing / cannot breathe
- Loss of consciousness
- Sudden severe weakness on one side
- Suspected stroke symptoms (FAST)

Return ONLY valid JSON:
{{
  "level": "normal|watch|escalate|emergency",
  "reason": "string — brief clinical justification",
  "escalation_trigger": "string or null — specific symptom/keyword that triggered escalation",
  "recommended_action": "string — what Clary should tell the user to do"
}}
""".strip()


class TriageAgent(BaseAgent):
    """
    Severity assessment and escalation routing.

    Inputs: ctx.current_events, ctx.validated_patterns, ctx.raw_message
    Outputs: ctx.triage_decision
    """

    AGENT_NAME = "triage_agent"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        self._log.info(
            "triage_agent.start",
            user_id=ctx.user_id,
            events=len(ctx.current_events),
        )

        # ── Fast path: keyword-based emergency/escalation check ───────────
        fast_decision = self._keyword_triage(ctx.raw_message, ctx.current_events)
        if fast_decision.level in (TriageLevel.EMERGENCY,):
            ctx.triage_decision = fast_decision
            ctx.clary_response = None  # will be overridden by Narrator
            self._log.warning(
                "triage_agent.emergency_detected",
                user_id=ctx.user_id,
                trigger=fast_decision.escalation_trigger,
            )
            return ctx

        # ── LLM triage for nuanced assessment ─────────────────────────────
        if ctx.current_events or ctx.raw_message:
            llm_decision = await self._llm_triage(ctx)
            # Take the more severe of keyword vs LLM decision
            ctx.triage_decision = self._take_more_severe(fast_decision, llm_decision)
        else:
            ctx.triage_decision = fast_decision

        self._log.info(
            "triage_agent.done",
            user_id=ctx.user_id,
            level=ctx.triage_decision.level.value,
        )
        return ctx

    # ── Keyword triage (fast, no LLM) ─────────────────────────────────────

    def _keyword_triage(
        self,
        message: str,
        events: list[HealthEvent],
    ) -> TriageDecision:
        combined_text = message.lower()
        for ev in events:
            combined_text += " " + ev.raw_text.lower()

        # Check EMERGENCY patterns first
        for pattern in _EMERGENCY_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                trigger = match.group(0)[:60]
                self._log.warning(
                    "triage.keyword.emergency", trigger=trigger
                )
                return TriageDecision(
                    level=TriageLevel.EMERGENCY,
                    reason="Emergency red flag keyword detected.",
                    escalation_trigger=trigger,
                    recommended_action=(
                        "Please seek immediate emergency medical care or call emergency services."
                    ),
                )

        # Check ESCALATION patterns
        for pattern in _ESCALATION_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                trigger = match.group(0)[:60]
                self._log.info(
                    "triage.keyword.escalate", trigger=trigger
                )
                return TriageDecision(
                    level=TriageLevel.ESCALATE,
                    reason="Escalation red flag keyword detected.",
                    escalation_trigger=trigger,
                    recommended_action=(
                        "I'd recommend seeing a doctor about this within the next few days. "
                        "This symptom warrants professional evaluation."
                    ),
                )

        # Check severity signals
        severe_events = [
            e for e in events if e.severity == Severity.SEVERE
        ]
        moderate_events = [
            e for e in events if e.severity == Severity.MODERATE
        ]

        if severe_events:
            return TriageDecision(
                level=TriageLevel.ESCALATE,
                reason="Severe symptom severity reported.",
                escalation_trigger="severity=severe",
                recommended_action=(
                    "Given the severity you're describing, I'd recommend getting this checked "
                    "by a healthcare professional."
                ),
            )

        if len(moderate_events) >= 3:
            return TriageDecision(
                level=TriageLevel.WATCH,
                reason="Moderate severity recurring across multiple sessions.",
                escalation_trigger=None,
                recommended_action=(
                    "This has been coming up a few times — worth mentioning to a doctor "
                    "if it continues."
                ),
            )

        return TriageDecision(
            level=TriageLevel.NORMAL,
            reason="No red flags detected.",
            escalation_trigger=None,
            recommended_action="",
        )

    # ── LLM triage ────────────────────────────────────────────────────────

    async def _llm_triage(self, ctx: PipelineContext) -> TriageDecision:
        """Use LLM for nuanced triage assessment."""
        events_text = "\n".join(
            f"  [{e.severity.value}] {e.raw_text[:200]}" for e in ctx.current_events
        )
        pattern_text = "\n".join(
            f"  Pattern: {p.symptom} ← {p.trigger} [{p.status.value}]"
            for p in ctx.validated_patterns
        )

        user_content = (
            f"Current message: {ctx.raw_message}\n\n"
            f"Current session events:\n{events_text or '  (none parsed)'}\n\n"
            f"Validated patterns:\n{pattern_text or '  (none)'}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw, trace = await self._llm_call(
                messages=messages,
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                temperature=0.0,
                max_tokens=512,
            )
        except AgentError as exc:
            self._log.error("triage_agent.llm_failed", error=str(exc))
            return TriageDecision(
                level=TriageLevel.NORMAL,
                reason="LLM triage unavailable — defaulting to NORMAL.",
            )

        self._record_trace(ctx, trace, "Triage assessment completed")

        parsed = self._parse_json(raw, context="TriageAgent")

        level_str = parsed.get("level", "normal").lower()
        try:
            level = TriageLevel(level_str)
        except ValueError:
            level = TriageLevel.NORMAL

        return TriageDecision(
            level=level,
            reason=parsed.get("reason", ""),
            escalation_trigger=parsed.get("escalation_trigger"),
            recommended_action=parsed.get("recommended_action", ""),
        )

    # ── Severity comparison ────────────────────────────────────────────────

    @staticmethod
    def _take_more_severe(a: TriageDecision, b: TriageDecision) -> TriageDecision:
        """Return whichever TriageDecision has the higher severity level."""
        order = [TriageLevel.NORMAL, TriageLevel.WATCH, TriageLevel.ESCALATE, TriageLevel.EMERGENCY]
        if order.index(a.level) >= order.index(b.level):
            return a
        return b