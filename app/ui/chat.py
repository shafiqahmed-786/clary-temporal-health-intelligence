"""
app/ui/chat.py — Chat Interface

Features:
  - Streamed assistant responses (OpenAI streaming via generator)
  - Persistent conversational history with timestamps
  - Clarification question detection and styled display
  - Pattern alert cards within chat flow
  - Triage escalation banners
  - JSON reasoning trace expanders (toggle-gated)
  - Typing indicator animation
"""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from app.state import ClaryState


# ── Helpers ────────────────────────────────────────────────────────────────────

def _confidence_badge(level: str) -> str:
    css = {
        "very_high": "badge-teal",
        "high": "badge-blue",
        "moderate": "badge-amber",
        "low": "badge-coral",
        "rejected": "badge-coral",
    }.get(level, "badge-blue")
    return f'<span class="badge {css}">{level.upper().replace("_"," ")}</span>'


def _status_badge(status: str) -> str:
    css = {
        "confirmed": "badge-teal",
        "emerging": "badge-amber",
        "watching": "badge-purple",
        "rejected": "badge-coral",
    }.get(status, "badge-blue")
    icon = {"confirmed": "✓", "emerging": "◐", "watching": "○", "rejected": "✗"}.get(status, "·")
    return f'<span class="badge {css}">{icon} {status.upper()}</span>'


def _triage_banner(level: str, action: str = "") -> None:
    if level == "emergency":
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);
             border-radius:8px;padding:14px 16px;margin:8px 0;">
            <div style="color:#f87171;font-weight:700;font-size:14px;">🚨 EMERGENCY — Seek Immediate Medical Attention</div>
            <div style="color:#fca5a5;font-size:13px;margin-top:4px;">{action}</div>
        </div>
        """, unsafe_allow_html=True)
    elif level == "escalate":
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
             border-radius:8px;padding:12px 16px;margin:8px 0;">
            <div style="color:#fbbf24;font-weight:600;font-size:13px;">⚠️ Medical Attention Recommended</div>
            <div style="color:#fcd34d;font-size:12px;margin-top:4px;">{action}</div>
        </div>
        """, unsafe_allow_html=True)


def _pattern_alert_card(pattern: dict) -> None:
    """Inline pattern alert card shown within the chat flow."""
    status = pattern.get("status", "watching")
    confidence = pattern.get("confidence", "low")
    symptom = pattern.get("symptom", "—")
    trigger = pattern.get("trigger", "—")
    n = pattern.get("occurrence_count", 0)
    lag_min = pattern.get("lag_days_min", 0)
    lag_max = pattern.get("lag_days_max", 0)
    mechanism = pattern.get("lag_registry_match", {})
    mechanism_name = mechanism.get("mechanism_name", "") if mechanism else ""
    evidence = pattern.get("evidence", [])

    status_css = {"confirmed": "confirmed", "emerging": "emerging", "watching": "watching"}.get(status, "watching")

    lag_str = (
        "same day" if lag_max <= 1
        else f"{lag_min}–{lag_max} days" if lag_max > 0
        else "unknown lag"
    )

    evidence_dates = [
        e.get("occurred_at", "")[:10] for e in evidence[:4] if e.get("occurred_at")
    ]

    st.markdown(f"""
    <div class="pattern-card {status_css}" style="margin:10px 0 6px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="font-size:15px;">🔍</div>
            <div class="pattern-title">{trigger.title()} → {symptom.title()}</div>
            <div style="margin-left:auto;display:flex;gap:6px;">
                {_status_badge(status)} {_confidence_badge(confidence)}
            </div>
        </div>
        <div class="pattern-meta">
            <span>N = {n} occurrences</span> &nbsp;·&nbsp;
            <span>Lag: {lag_str}</span>
            {f"&nbsp;·&nbsp;<span>Mechanism: {mechanism_name}</span>" if mechanism_name else ""}
        </div>
        {f'<div class="pattern-meta" style="margin-top:4px;">Evidence dates: {", ".join(evidence_dates)}</div>' if evidence_dates else ""}
    </div>
    """, unsafe_allow_html=True)


def _render_action_items(items: list[str]) -> None:
    if not items:
        return
    st.markdown('<div style="margin-top:10px;">', unsafe_allow_html=True)
    for item in items:
        st.markdown(
            f'<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:4px;">'
            f'<span style="color:#3b82f6;margin-top:2px;">→</span>'
            f'<span style="font-size:13px;color:#94a3b8;">{item}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_trace_expander(traces: list[dict]) -> None:
    if not traces or not ClaryState.show_traces():
        return
    with st.expander("🔬 Reasoning Trace", expanded=False):
        for trace in traces:
            agent = trace.get("agent", "?")
            data = trace.get("data", {})
            ts = trace.get("ts", "")
            st.markdown(f'<div style="font-size:10px;color:#6b7280;font-family:monospace;margin-bottom:6px;">{agent} · {ts[:19]}</div>', unsafe_allow_html=True)
            st.json(data, expanded=False)


# ── Message rendering ──────────────────────────────────────────────────────────

def _render_user_message(msg: dict) -> None:
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg["content"])
        ts = msg.get("timestamp", "")[:19].replace("T", " ")
        st.markdown(f'<div class="message-meta">{ts}</div>', unsafe_allow_html=True)


def _render_assistant_message(msg: dict) -> None:
    with st.chat_message("assistant", avatar="⬡"):
        # Triage banner (if escalation)
        triage = msg.get("triage_level", "normal")
        action = ""
        if msg.get("escalate") and msg.get("action_items"):
            action = msg["action_items"][0]
        if triage in ("escalate", "emergency"):
            _triage_banner(triage, action)

        # Main response text
        st.markdown(msg["content"])

        # Action items
        non_escalation_actions = [a for a in msg.get("action_items", []) if a != action]
        _render_action_items(non_escalation_actions)

        # Pattern cards
        for pattern in msg.get("patterns", []):
            _pattern_alert_card(pattern)

        # Metadata row
        ts = msg.get("timestamp", "")[:19].replace("T", " ")
        tone = msg.get("tone", "")
        tone_icons = {
            "empathetic": "💙", "informative": "ℹ️",
            "pattern_alert": "🔍", "escalation": "⚠️", "clarifying": "❓"
        }
        tone_icon = tone_icons.get(tone, "")
        st.markdown(
            f'<div class="message-meta">{ts} &nbsp;·&nbsp; {tone_icon} {tone}</div>',
            unsafe_allow_html=True,
        )

        # Reasoning traces
        _render_trace_expander(msg.get("traces", []))

        # Disclaimer (collapsible, only when patterns cited)
        if msg.get("patterns"):
            with st.expander("📋 Disclaimer", expanded=False):
                st.caption(
                    "Clary provides health observations, not medical diagnoses. "
                    "Always consult a qualified healthcare professional for medical advice."
                )


# ── Clarification UI ───────────────────────────────────────────────────────────

def _render_clarification_hint() -> None:
    q = ClaryState.get_clarification_question()
    if q and ClaryState.is_awaiting_clarification():
        st.markdown(f"""
        <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);
             border-radius:8px;padding:10px 14px;margin:8px 0;">
            <span style="color:#a78bfa;font-size:12px;">❓ Clary is asking: </span>
            <span style="color:#ddd6fe;font-size:13px;">{q}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Streaming response ─────────────────────────────────────────────────────────

def _stream_response(text: str):
    """Generator that yields the response token-by-token for streaming display."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.018)


# ── Welcome screen ─────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 40px;">
        <div style="font-size:48px;margin-bottom:16px;">⬡</div>
        <div style="font-size:24px;font-weight:600;color:#f1f5f9;letter-spacing:-0.5px;margin-bottom:8px;">
            Hello, I'm Clary.
        </div>
        <div style="font-size:15px;color:#6b7280;max-width:480px;margin:0 auto;line-height:1.7;">
            Your personal health companion. I track patterns in your health over time —
            connecting symptoms to triggers that may have occurred days or weeks earlier.
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;max-width:700px;margin:0 auto 40px;">
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:22px;margin-bottom:6px;">🔍</div>
            <div style="font-size:12px;font-weight:600;color:#f1f5f9;">Pattern Detection</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px;">Finds recurring health signals across sessions</div>
        </div>
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:22px;margin-bottom:6px;">⏳</div>
            <div style="font-size:12px;font-weight:600;color:#f1f5f9;">Temporal Reasoning</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px;">Connects triggers to symptoms weeks later</div>
        </div>
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:22px;margin-bottom:6px;">🧐</div>
            <div style="font-size:12px;font-weight:600;color:#f1f5f9;">Skeptic Validated</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px;">Patterns only confirmed with rigorous evidence</div>
        </div>
    </div>

    <div style="text-align:center;font-size:13px;color:#4b5563;">
        Try: <em>"I have a bad stomach ache again"</em> · <em>"My hair is falling out lately"</em>
    </div>
    """, unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    """Main chat page renderer."""
    ClaryState.init()

    # Header
    col_h1, col_h2 = st.columns([7, 3])
    with col_h1:
        user_id = ClaryState.get_user_id()
        st.markdown(f'<h2 style="color:#f1f5f9;font-weight:600;margin:0;">💬 Chat</h2>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#4b5563;margin-top:2px;font-family:monospace;">{user_id} · {ClaryState.get_session_id()[:20]}…</div>', unsafe_allow_html=True)
    with col_h2:
        n_patterns = len(ClaryState.get_patterns())
        n_msgs = len(ClaryState.get_messages())
        st.markdown(
            f'<div style="text-align:right;margin-top:4px;">'
            f'<span class="badge badge-blue">{n_patterns} pattern{"s" if n_patterns != 1 else ""}</span> &nbsp;'
            f'<span class="badge badge-purple">{n_msgs} message{"s" if n_msgs != 1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Message history
    messages = ClaryState.get_messages()

    if not messages:
        _render_welcome()
    else:
        msg_container = st.container()
        with msg_container:
            for msg in messages:
                if msg["role"] == "user":
                    _render_user_message(msg)
                else:
                    _render_assistant_message(msg)

    # Clarification hint
    _render_clarification_hint()

    # ── Chat input ─────────────────────────────────────────────────────────
    placeholder = (
        "Answer Clary's question above…"
        if ClaryState.is_awaiting_clarification()
        else "Tell me how you're feeling…"
    )

    if user_input := st.chat_input(placeholder):
        # Immediately show the user message
        ClaryState.append_message(role="user", content=user_input)

        # Show typing indicator
        with st.chat_message("assistant", avatar="⬡"):
            with st.spinner("Clary is thinking…"):
                # Send to backend
                response = ClaryState.send_message(user_input)

            if response is None:
                st.error("Something went wrong. Please try again.")
                st.rerun()

            # Detect if response is a clarifying question
            tone = response.get("tone", "")
            is_clarification = tone == "clarifying"

            if is_clarification:
                ClaryState.set_clarification(True, response["text"])
            else:
                ClaryState.set_clarification(False, None)

            # Stream the response
            response_text = response.get("text", "")
            streamed = st.write_stream(_stream_response(response_text))

            # Render action items immediately
            _render_action_items(response.get("action_items", []))

            # Get updated patterns from state (populated by orchestrator)
            current_patterns = ClaryState.get_patterns()
            new_patterns = current_patterns[-3:]  # Most recent patterns for this response

            # Render pattern cards for new patterns
            for p in new_patterns:
                if p.get("status") in ("confirmed", "emerging"):
                    _pattern_alert_card(p)

            # Triage banner
            triage = response.get("triage_level", "normal")
            if triage in ("escalate", "emergency"):
                action = response.get("action_items", [""])[0]
                _triage_banner(triage, action)

        # Persist to state
        ClaryState.append_message(
            role="assistant",
            content=response_text,
            patterns=new_patterns,
            triage_level=triage,
            action_items=response.get("action_items", []),
            escalate=response.get("escalate", False),
        )
        ClaryState.set_triage_level(triage)

        # Refresh timeline data in background
        ClaryState.refresh_timeline()

        st.rerun()


# ── Module alias (for pages/ auto-discovery) ───────────────────────────────────
__all__ = ["render"]