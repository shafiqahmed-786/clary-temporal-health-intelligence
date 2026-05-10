"""
app/main.py — Clary Streamlit Entry Point

Run with:
    streamlit run app/main.py

Responsibilities:
  - Inject global CSS (dark premium theme)
  - Render sidebar navigation + user selector
  - Route to the correct page module
  - Show global error/status banners
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# ── Path resolution ────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
for p in [_ROOT, _APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.state import ClaryState  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clary · Health Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": "Clary v1.0"},
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Dark background override */
.stApp { background: #0a0c12; }
section[data-testid="stSidebar"] { background: #0f1117 !important; border-right: 1px solid #1f2937; }
.stChatMessage { background: transparent !important; }

/* ── Typography ── */
h1, h2, h3 { letter-spacing: -0.5px; }

/* ── Sidebar nav ── */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border-radius: 8px;
    color: #94a3b8; text-decoration: none;
    font-size: 13px; font-weight: 500;
    margin-bottom: 2px; cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid transparent;
}
.nav-item:hover { background: #1f2937; color: #f1f5f9; }
.nav-item.active { background: rgba(59,130,246,0.12); color: #3b82f6; border-color: rgba(59,130,246,0.25); }
.nav-icon { font-size: 16px; min-width: 20px; text-align: center; }

/* ── Cards ── */
.clary-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 20px 22px; margin-bottom: 14px;
}
.clary-card:hover { border-color: #374151; transition: border-color 0.2s; }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 100px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
    font-family: 'JetBrains Mono', monospace;
}
.badge-blue   { background: rgba(59,130,246,0.15);  color: #60a5fa; }
.badge-teal   { background: rgba(20,184,166,0.15);  color: #2dd4bf; }
.badge-amber  { background: rgba(245,158,11,0.15);  color: #fbbf24; }
.badge-coral  { background: rgba(239,68,68,0.15);   color: #f87171; }
.badge-purple { background: rgba(139,92,246,0.15);  color: #a78bfa; }
.badge-green  { background: rgba(16,185,129,0.15);  color: #34d399; }

/* ── Chat ── */
.user-bubble {
    background: #1d4ed8; border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; max-width: 80%; margin-left: auto;
    color: #eff6ff; font-size: 14px; line-height: 1.6;
}
.assistant-bubble {
    background: #1f2937; border: 1px solid #374151;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; max-width: 82%;
    color: #f1f5f9; font-size: 14px; line-height: 1.6;
}
.message-meta {
    font-size: 10px; color: #4b5563;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}

/* ── Pattern cards ── */
.pattern-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 18px 20px;
    margin-bottom: 12px; position: relative;
}
.pattern-card.confirmed { border-left: 3px solid #3b82f6; }
.pattern-card.emerging  { border-left: 3px solid #f59e0b; }
.pattern-card.watching  { border-left: 3px solid #6b7280; }
.pattern-title { font-size: 14px; font-weight: 600; color: #f1f5f9; margin-bottom: 6px; }
.pattern-meta  { font-size: 12px; color: #6b7280; }

/* ── Metrics ── */
.metric-box {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 16px 18px; text-align: center;
}
.metric-val   { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 11px; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Escalation banner ── */
.escalation-banner {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px; padding: 12px 16px; color: #fca5a5;
    font-size: 13px; margin: 8px 0;
}

/* ── Triage indicators ── */
.triage-normal   { color: #34d399; }
.triage-watch    { color: #fbbf24; }
.triage-escalate { color: #f87171; }
.triage-emergency { color: #ef4444; font-weight: 700; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: #1f2937; border: 1px solid #374151;
    color: #f1f5f9; border-radius: 8px; font-weight: 500;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #374151; border-color: #4b5563;
}
.stButton > button.primary {
    background: #2563eb; border-color: #3b82f6; color: white;
}
div[data-testid="stChatInput"] textarea {
    background: #111827 !important; border: 1px solid #374151 !important;
    color: #f1f5f9 !important; border-radius: 12px !important;
}
.stSelectbox > div > div {
    background: #111827 !important; border-color: #374151 !important;
}
.stExpander { background: #111827 !important; border: 1px solid #1f2937 !important; border-radius: 8px !important; }
hr { border-color: #1f2937 !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #111827; border: 1px solid #1f2937;
    color: #94a3b8; border-radius: 8px; padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.12) !important;
    color: #60a5fa !important; border-color: rgba(59,130,246,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Initialise state ───────────────────────────────────────────────────────────
ClaryState.init()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _sidebar() -> str:
    """Render sidebar and return the active page key."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="padding: 4px 0 24px; border-bottom: 1px solid #1f2937; margin-bottom: 20px;">
            <div style="font-size:20px;font-weight:700;letter-spacing:-0.5px;
                        background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                ⬡ Clary
            </div>
            <div style="font-size:10px;color:#4b5563;text-transform:uppercase;
                        letter-spacing:1.5px;font-family:'JetBrains Mono',monospace;margin-top:2px;">
                Health Intelligence v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        # User selector
        st.markdown('<div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Active User</div>', unsafe_allow_html=True)
        user_options = {
            "USR001 — Arjun Sharma": "USR001",
            "USR002 — Meera Nair": "USR002",
            "USR003 — Priya Pillai": "USR003",
        }
        selected_label = st.selectbox(
            "user_selector",
            options=list(user_options.keys()),
            label_visibility="collapsed",
            key="sidebar_user_select",
        )
        new_uid = user_options[selected_label]
        if new_uid != ClaryState.get_user_id():
            ClaryState.set_user_id(new_uid)
            st.rerun()

        st.markdown("<div style='margin: 16px 0 8px; font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;'>Navigation</div>", unsafe_allow_html=True)

        pages = [
            ("chat",     "💬", "Chat",          "Conversation interface"),
            ("timeline", "📅", "Timeline",       "Health event history"),
            ("patterns", "🔍", "Patterns",       "Detected patterns"),
            ("graph",    "🕸️", "Event Graph",    "Causal network"),
            ("eval",     "📊", "Evaluation",     "Model metrics"),
        ]

        active = ClaryState.get_active_page()
        new_page = active

        for key, icon, label, desc in pages:
            is_active = active == key
            css_class = "nav-item active" if is_active else "nav-item"
            clicked = st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="secondary",
            )
            if clicked:
                new_page = key

        if new_page != active:
            ClaryState.set_active_page(new_page)
            st.rerun()

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.divider()

        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            show_traces = st.toggle(
                "Show reasoning traces",
                value=ClaryState.show_traces(),
                key="toggle_traces",
            )
            if show_traces != ClaryState.show_traces():
                ClaryState.toggle_traces()

            if st.button("🗑️ Clear chat history", use_container_width=True):
                ClaryState.clear_messages()
                st.rerun()

        # Status indicators
        triage = ClaryState.get_triage_level()
        triage_css = f"triage-{triage}"
        triage_icons = {"normal": "🟢", "watch": "🟡", "escalate": "🔴", "emergency": "🚨"}
        st.markdown(
            f'<div style="font-size:11px;color:#4b5563;margin-top:12px;">'
            f'Session status: <span class="{triage_css}">'
            f'{triage_icons.get(triage,"⚪")} {triage.upper()}</span></div>',
            unsafe_allow_html=True,
        )
        uid = ClaryState.get_user_id()
        st.markdown(f'<div style="font-size:10px;color:#374151;font-family:monospace;margin-top:4px;">{uid}</div>', unsafe_allow_html=True)

        return ClaryState.get_active_page()


# ── Error banner ───────────────────────────────────────────────────────────────
def _render_error_banner() -> None:
    err = ClaryState.get_error()
    if err:
        col1, col2 = st.columns([10, 1])
        with col1:
            st.error(f"⚠️ {err}")
        with col2:
            if st.button("✕", key="clear_error"):
                ClaryState.set_error(None)
                st.rerun()


# ── Page routing ───────────────────────────────────────────────────────────────
def main() -> None:
    active_page = _sidebar()
    _render_error_banner()

    if active_page == "chat":
        from app.ui import chat as _chat
        _chat.render()
    elif active_page == "timeline":
        from app.ui import timeline as _timeline
        _timeline.render()
    elif active_page == "patterns":
        from app.ui import patterns as _patterns
        _patterns.render()
    elif active_page == "graph":
        from app.ui import graph as _graph
        _graph.render()
    elif active_page == "eval":
        from app.ui import eval_dashboard as _eval
        _eval.render()


if __name__ == "__main__":
    main()