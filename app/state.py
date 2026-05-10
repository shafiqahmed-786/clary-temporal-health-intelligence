"""
app/state.py — ClaryState

Single source of truth for all Streamlit session state.
Every page reads and writes through this class — never touching
st.session_state directly. This enforces a clean contract between
pages and prevents key-name drift across a multi-page app.

Initialization:
    ClaryState.init()           # call at the top of every page

Reading:
    ClaryState.get_messages()
    ClaryState.get_user_id()

Writing:
    ClaryState.set_user_id("USR001")
    ClaryState.append_message(role, content, meta)

The Orchestrator instance is cached via @st.cache_resource and accessed
through ClaryState.get_orchestrator() — never instantiated per-render.
"""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime
from typing import Any

import streamlit as st

# ── Path fix: allow imports from project root ──────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Async bridge ───────────────────────────────────────────────────────────────

def run_async(coro) -> Any:
    """
    Execute an async coroutine synchronously.
    Streamlit runs in a synchronous context; all async backend calls
    go through this bridge.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside a running loop (shouldn't happen in Streamlit main thread)
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ── Orchestrator cache ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising Clary engine…")
def _get_orchestrator_cached():
    """
    Cache the Orchestrator singleton for the lifetime of the Streamlit server.
    All users share this instance (Neo4j + ChromaDB pools are thread-safe).
    """
    try:
        from agents.orchestrator import Orchestrator
        from graph.event_graph import EventGraph
        graph = EventGraph()
        return Orchestrator.create(event_graph=graph)
    except Exception as exc:
        st.error(f"Backend initialisation failed: {exc}")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _cached_patterns(user_id: str) -> list[dict]:
    """Cache validated patterns for 60 seconds to avoid re-querying."""
    try:
        from memory.episodic_store import EpisodicStore
        store = EpisodicStore()
        # Fetch from working memory or episodic store
        return []
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner=False)
def _cached_timeline_events(user_id: str) -> list[dict]:
    """Cache timeline events for 2 minutes."""
    try:
        from memory.episodic_store import EpisodicStore
        store = EpisodicStore()
        raw = run_async(store.get_all_events(user_id))
        return raw
    except Exception:
        return []


# ── State keys (single source of truth) ───────────────────────────────────────

class _K:
    """All st.session_state keys in one place."""
    INITIALISED     = "_clary_initialised"
    USER_ID         = "user_id"
    SESSION_ID      = "session_id"
    MESSAGES        = "messages"           # list[dict] — chat history
    PATTERNS        = "patterns"           # list[TemporalPattern dicts]
    TIMELINE_EVENTS = "timeline_events"
    ACTIVE_PAGE     = "active_page"
    AWAITING_CLARIF = "awaiting_clarification"
    CLARIF_Q        = "clarification_question"
    TRIAGE_LEVEL    = "triage_level"
    SHOW_TRACES     = "show_reasoning_traces"
    GRAPH_DATA      = "graph_data"
    EVAL_RESULTS    = "eval_results"
    LAST_RESPONSE   = "last_clary_response"
    ERROR_BANNER    = "error_banner"
    TYPING          = "typing_indicator"


# ── ClaryState ─────────────────────────────────────────────────────────────────

class ClaryState:
    """Namespace class — all methods are @staticmethod."""

    @staticmethod
    def init(default_user_id: str = "USR001") -> None:
        """Idempotent initialisation — safe to call at top of every page."""
        ss = st.session_state
        if ss.get(_K.INITIALISED):
            return

        ss[_K.USER_ID]         = default_user_id
        ss[_K.SESSION_ID]      = f"{default_user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        ss[_K.MESSAGES]        = []
        ss[_K.PATTERNS]        = []
        ss[_K.TIMELINE_EVENTS] = []
        ss[_K.ACTIVE_PAGE]     = "chat"
        ss[_K.AWAITING_CLARIF] = False
        ss[_K.CLARIF_Q]        = None
        ss[_K.TRIAGE_LEVEL]    = "normal"
        ss[_K.SHOW_TRACES]     = False
        ss[_K.GRAPH_DATA]      = None
        ss[_K.EVAL_RESULTS]    = None
        ss[_K.LAST_RESPONSE]   = None
        ss[_K.ERROR_BANNER]    = None
        ss[_K.TYPING]          = False
        ss[_K.INITIALISED]     = True

    # ── User / Session ─────────────────────────────────────────────────────

    @staticmethod
    def get_user_id() -> str:
        return st.session_state.get(_K.USER_ID, "USR001")

    @staticmethod
    def set_user_id(user_id: str) -> None:
        st.session_state[_K.USER_ID] = user_id
        # Reset session when user changes
        st.session_state[_K.SESSION_ID] = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        st.session_state[_K.MESSAGES] = []
        st.session_state[_K.PATTERNS] = []
        st.session_state[_K.TIMELINE_EVENTS] = []
        st.session_state[_K.GRAPH_DATA] = None
        _cached_timeline_events.clear()
        _cached_patterns.clear()

    @staticmethod
    def get_session_id() -> str:
        return st.session_state.get(_K.SESSION_ID, "")

    # ── Messages ───────────────────────────────────────────────────────────

    @staticmethod
    def get_messages() -> list[dict]:
        return st.session_state.get(_K.MESSAGES, [])

    @staticmethod
    def append_message(
        role: str,
        content: str,
        patterns: list[dict] | None = None,
        traces: list[dict] | None = None,
        triage_level: str = "normal",
        action_items: list[str] | None = None,
        escalate: bool = False,
    ) -> None:
        st.session_state[_K.MESSAGES].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "patterns": patterns or [],
            "traces": traces or [],
            "triage_level": triage_level,
            "action_items": action_items or [],
            "escalate": escalate,
        })

    @staticmethod
    def clear_messages() -> None:
        st.session_state[_K.MESSAGES] = []

    # ── Patterns ───────────────────────────────────────────────────────────

    @staticmethod
    def get_patterns() -> list[dict]:
        return st.session_state.get(_K.PATTERNS, [])

    @staticmethod
    def set_patterns(patterns: list[dict]) -> None:
        st.session_state[_K.PATTERNS] = patterns

    @staticmethod
    def upsert_pattern(pattern: dict) -> None:
        """Update existing or append new pattern."""
        patterns = ClaryState.get_patterns()
        pid = pattern.get("pattern_id")
        for i, p in enumerate(patterns):
            if p.get("pattern_id") == pid:
                patterns[i] = pattern
                st.session_state[_K.PATTERNS] = patterns
                return
        patterns.append(pattern)
        st.session_state[_K.PATTERNS] = patterns

    # ── Timeline ───────────────────────────────────────────────────────────

    @staticmethod
    def get_timeline_events() -> list[dict]:
        return st.session_state.get(_K.TIMELINE_EVENTS, [])

    @staticmethod
    def set_timeline_events(events: list[dict]) -> None:
        st.session_state[_K.TIMELINE_EVENTS] = events

    # ── UI flags ───────────────────────────────────────────────────────────

    @staticmethod
    def get_active_page() -> str:
        return st.session_state.get(_K.ACTIVE_PAGE, "chat")

    @staticmethod
    def set_active_page(page: str) -> None:
        st.session_state[_K.ACTIVE_PAGE] = page

    @staticmethod
    def is_awaiting_clarification() -> bool:
        return st.session_state.get(_K.AWAITING_CLARIF, False)

    @staticmethod
    def set_clarification(pending: bool, question: str | None = None) -> None:
        st.session_state[_K.AWAITING_CLARIF] = pending
        st.session_state[_K.CLARIF_Q] = question

    @staticmethod
    def get_clarification_question() -> str | None:
        return st.session_state.get(_K.CLARIF_Q)

    @staticmethod
    def get_triage_level() -> str:
        return st.session_state.get(_K.TRIAGE_LEVEL, "normal")

    @staticmethod
    def set_triage_level(level: str) -> None:
        st.session_state[_K.TRIAGE_LEVEL] = level

    @staticmethod
    def show_traces() -> bool:
        return st.session_state.get(_K.SHOW_TRACES, False)

    @staticmethod
    def toggle_traces() -> None:
        st.session_state[_K.SHOW_TRACES] = not st.session_state.get(_K.SHOW_TRACES, False)

    @staticmethod
    def set_typing(value: bool) -> None:
        st.session_state[_K.TYPING] = value

    @staticmethod
    def is_typing() -> bool:
        return st.session_state.get(_K.TYPING, False)

    # ── Graph ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_graph_data() -> dict | None:
        return st.session_state.get(_K.GRAPH_DATA)

    @staticmethod
    def set_graph_data(data: dict) -> None:
        st.session_state[_K.GRAPH_DATA] = data

    # ── Eval ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_eval_results() -> dict | None:
        return st.session_state.get(_K.EVAL_RESULTS)

    @staticmethod
    def set_eval_results(results: dict) -> None:
        st.session_state[_K.EVAL_RESULTS] = results

    # ── Error / status ─────────────────────────────────────────────────────

    @staticmethod
    def set_error(msg: str | None) -> None:
        st.session_state[_K.ERROR_BANNER] = msg

    @staticmethod
    def get_error() -> str | None:
        return st.session_state.get(_K.ERROR_BANNER)

    # ── Orchestrator access ────────────────────────────────────────────────

    @staticmethod
    def get_orchestrator():
        return _get_orchestrator_cached()

    @staticmethod
    def send_message(user_text: str) -> dict | None:
        """
        Send a message through the Orchestrator and return the ClaryResponse dict.
        Handles the sync/async bridge and updates state.
        """
        orch = ClaryState.get_orchestrator()
        if orch is None:
            ClaryState.set_error("Backend not available. Please check configuration.")
            return None

        user_id = ClaryState.get_user_id()
        session_id = ClaryState.get_session_id()

        try:
            response = run_async(
                orch.process_message(
                    user_id=user_id,
                    message=user_text,
                    session_id=session_id,
                )
            )
            if response is None:
                return None

            # Update state from response
            ClaryState.set_triage_level(response.triage.level.value)

            # Convert patterns to dicts for serialisation
            if hasattr(orch, "_skeptic_agent"):  # internal state not exposed
                pass

            return {
                "text": response.text,
                "tone": response.tone.value,
                "triage_level": response.triage.level.value,
                "action_items": response.action_items,
                "escalate": response.escalate,
                "disclaimer": response.disclaimer,
            }
        except Exception as exc:
            ClaryState.set_error(f"Processing error: {exc}")
            return None

    @staticmethod
    def refresh_timeline() -> None:
        """Force-refresh timeline from episodic store."""
        user_id = ClaryState.get_user_id()
        _cached_timeline_events.clear()
        events = run_async(_fetch_timeline(user_id))
        ClaryState.set_timeline_events(events)


async def _fetch_timeline(user_id: str) -> list[dict]:
    try:
        from memory.episodic_store import EpisodicStore
        store = EpisodicStore()
        return await store.get_all_events(user_id)
    except Exception:
        return []