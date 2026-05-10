"""
app/ui/timeline.py — Health Event Timeline

Visualises the user's health history chronologically.

Charts:
  1. Event scatter plot — symptoms plotted on a timeline, coloured by severity
  2. Symptom frequency heatmap — which symptoms appear in which weeks
  3. Lag annotation overlay — arrows from trigger dates to symptom dates
  4. Pattern windows — shaded regions showing confirmed trigger → effect windows
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.state import ClaryState, run_async


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_events(user_id: str) -> list[dict]:
    """Load raw events from episodic store (cached via state)."""
    events = ClaryState.get_timeline_events()
    if events:
        return events
    try:
        from memory.episodic_store import EpisodicStore
        store = EpisodicStore()
        raw = run_async(store.get_all_events(user_id))
        ClaryState.set_timeline_events(raw)
        return raw
    except Exception as e:
        st.warning(f"Could not load events from store: {e}. Using chat history.")
        return []


def _events_to_df(events: list[dict]) -> pd.DataFrame:
    """Convert raw event dicts to a flat DataFrame for plotting."""
    rows = []
    for ev in events:
        meta = ev.get("metadata", {})
        epoch = meta.get("timestamp_epoch", 0)
        try:
            dt = datetime.fromtimestamp(float(epoch)) if epoch else None
        except (ValueError, TypeError):
            dt = None

        if not dt:
            continue

        symptoms_raw = meta.get("symptoms", "") or ""
        triggers_raw = meta.get("triggers", "") or ""
        symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]
        triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()]
        severity = meta.get("severity", "unknown")

        # One row per symptom for the scatter plot
        if symptoms:
            for sym in symptoms:
                rows.append({
                    "date": dt,
                    "session_id": meta.get("session_id", ""),
                    "signal": sym,
                    "signal_type": "symptom",
                    "severity": severity,
                    "all_symptoms": ", ".join(symptoms),
                    "all_triggers": ", ".join(triggers),
                    "text": ev.get("document", "")[:120],
                })
        if triggers:
            for trig in triggers:
                rows.append({
                    "date": dt,
                    "session_id": meta.get("session_id", ""),
                    "signal": trig,
                    "signal_type": "trigger",
                    "severity": severity,
                    "all_symptoms": ", ".join(symptoms),
                    "all_triggers": ", ".join(triggers),
                    "text": ev.get("document", "")[:120],
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


def _severity_color(severity: str) -> str:
    return {
        "severe": "#ef4444",
        "moderate": "#f59e0b",
        "mild": "#3b82f6",
        "none": "#10b981",
        "unknown": "#6b7280",
    }.get(severity, "#6b7280")


# ── Chart 1: Event scatter timeline ───────────────────────────────────────────

def _render_event_scatter(df: pd.DataFrame, patterns: list[dict]) -> None:
    st.markdown("### Event Timeline")

    if df.empty:
        st.info("No events found for this user yet. Start a conversation to build your timeline.")
        return

    # Separate symptoms and triggers
    df_sym = df[df["signal_type"] == "symptom"].copy()
    df_trig = df[df["signal_type"] == "trigger"].copy()

    fig = go.Figure()

    # Trigger trace (diamonds, muted)
    if not df_trig.empty:
        fig.add_trace(go.Scatter(
            x=df_trig["date"],
            y=df_trig["signal"],
            mode="markers",
            name="Trigger / Behaviour",
            marker=dict(
                symbol="diamond",
                size=9,
                color="#4b5563",
                line=dict(color="#6b7280", width=1),
            ),
            text=df_trig.apply(
                lambda r: f"<b>{r['signal']}</b><br>Date: {r['date'].strftime('%b %d, %Y')}<br>Session: {r['session_id']}", axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Symptom trace (circles, coloured by severity)
    if not df_sym.empty:
        colors = df_sym["severity"].map(lambda s: _severity_color(s))
        fig.add_trace(go.Scatter(
            x=df_sym["date"],
            y=df_sym["signal"],
            mode="markers",
            name="Symptom",
            marker=dict(
                symbol="circle",
                size=12,
                color=colors,
                line=dict(color="#1f2937", width=2),
                opacity=0.9,
            ),
            text=df_sym.apply(
                lambda r: (
                    f"<b>{r['signal']}</b><br>"
                    f"Severity: {r['severity']}<br>"
                    f"Date: {r['date'].strftime('%b %d, %Y')}<br>"
                    f"Triggers: {r['all_triggers'] or 'none identified'}<br>"
                    f"Session: {r['session_id']}"
                ), axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Pattern lag annotations (arrows from trigger to symptom)
    for pat in patterns:
        if pat.get("status") not in ("confirmed", "emerging"):
            continue
        evidence = pat.get("evidence", [])
        for ev in evidence[:3]:
            trig_date = ev.get("trigger_observed_at") or ev.get("occurred_at")
            sym_date = ev.get("occurred_at")
            if not trig_date or not sym_date:
                continue
            try:
                t_dt = datetime.fromisoformat(trig_date[:19])
                s_dt = datetime.fromisoformat(sym_date[:19])
                if t_dt < s_dt:
                    fig.add_annotation(
                        x=s_dt, y=pat.get("symptom", ""),
                        ax=t_dt, ay=pat.get("symptom", ""),
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2, arrowsize=1, arrowwidth=1.5,
                        arrowcolor="#3b82f6",
                        opacity=0.6,
                    )
            except Exception:
                continue

    fig.update_layout(
        paper_bgcolor="#0a0c12",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=12),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h", y=-0.15,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor="#1f2937", linecolor="#374151",
            tickformat="%b %d",
        ),
        yaxis=dict(
            gridcolor="#1f2937", linecolor="#374151",
            autorange="reversed",
        ),
        height=420,
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Chart 2: Symptom frequency heatmap ────────────────────────────────────────

def _render_symptom_heatmap(df: pd.DataFrame) -> None:
    st.markdown("### Symptom Frequency Heatmap")

    df_sym = df[df["signal_type"] == "symptom"].copy()
    if df_sym.empty or len(df_sym) < 2:
        st.info("Not enough symptom data to build a heatmap yet.")
        return

    df_sym["week"] = df_sym["date"].dt.to_period("W").apply(lambda p: str(p.start_time.date()))
    pivot = df_sym.groupby(["signal", "week"]).size().unstack(fill_value=0)

    if pivot.empty:
        return

    # Keep top 8 symptoms by frequency
    top_symptoms = df_sym["signal"].value_counts().head(8).index.tolist()
    pivot = pivot.loc[[s for s in top_symptoms if s in pivot.index]]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[
            [0, "#111827"],
            [0.3, "#1e3a5f"],
            [0.6, "#2563eb"],
            [1, "#60a5fa"],
        ],
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Week of %{x}<br>Count: %{z}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color="#6b7280", size=10),
            bgcolor="#111827",
            bordercolor="#374151",
        ),
    ))

    fig.update_layout(
        paper_bgcolor="#0a0c12",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(gridcolor="#1f2937", tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1f2937"),
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Chart 3: Lag distribution per pattern ─────────────────────────────────────

def _render_lag_distribution(patterns: list[dict]) -> None:
    confirmed = [p for p in patterns if p.get("status") in ("confirmed", "emerging")]
    if not confirmed:
        return

    st.markdown("### Lag Window Distribution")
    st.caption("Days from trigger to symptom manifestation, per confirmed pattern")

    fig = go.Figure()

    for pat in confirmed[:5]:
        evidence = pat.get("evidence", [])
        lags = [e.get("lag_days", 0) for e in evidence if e.get("lag_days") is not None]
        if not lags:
            continue

        label = f"{pat.get('trigger','?')[:20]} → {pat.get('symptom','?')[:20]}"
        fig.add_trace(go.Box(
            y=lags,
            name=label,
            boxmean=True,
            marker_color="#3b82f6",
            line_color="#60a5fa",
        ))

    fig.update_layout(
        paper_bgcolor="#0a0c12",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=10, b=60),
        yaxis=dict(
            title="Lag (days)", gridcolor="#1f2937",
            linecolor="#374151",
        ),
        xaxis=dict(tickangle=-25),
        showlegend=False,
        height=280,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Summary stats ──────────────────────────────────────────────────────────────

def _render_summary_stats(df: pd.DataFrame, patterns: list[dict]) -> None:
    df_sym = df[df["signal_type"] == "symptom"] if not df.empty else pd.DataFrame()
    unique_syms = df_sym["signal"].nunique() if not df_sym.empty else 0
    total_sessions = df["session_id"].nunique() if not df.empty else 0
    confirmed_patterns = sum(1 for p in patterns if p.get("status") == "confirmed")
    span_days = 0
    if not df_sym.empty and len(df_sym) >= 2:
        span_days = (df_sym["date"].max() - df_sym["date"].min()).days

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, total_sessions, "Sessions", "#3b82f6"),
        (c2, unique_syms, "Unique Symptoms", "#8b5cf6"),
        (c3, confirmed_patterns, "Confirmed Patterns", "#10b981"),
        (c4, f"{span_days}d", "Timeline Span", "#f59e0b"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    ClaryState.init()
    user_id = ClaryState.get_user_id()

    st.markdown('<h2 style="color:#f1f5f9;font-weight:600;margin:0;">📅 Health Timeline</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:#4b5563;margin-top:2px;margin-bottom:16px;font-family:monospace;">{user_id}</div>', unsafe_allow_html=True)

    # Refresh button
    col_r1, col_r2 = st.columns([8, 2])
    with col_r2:
        if st.button("↺ Refresh", use_container_width=True):
            ClaryState.set_timeline_events([])
            st.rerun()

    # Load data
    with st.spinner("Loading timeline…"):
        raw_events = _load_events(user_id)
        df = _events_to_df(raw_events)
        patterns = ClaryState.get_patterns()

    st.divider()

    # Summary stats
    _render_summary_stats(df, patterns)

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # Date range filter
    if not df.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            start_date = st.date_input(
                "From",
                value=date_min,
                min_value=date_min,
                max_value=date_max,
                key="tl_start",
            )
        with col_f2:
            end_date = st.date_input(
                "To",
                value=date_max,
                min_value=date_min,
                max_value=date_max,
                key="tl_end",
            )

        # Filter dataframe
        df = df[
            (df["date"].dt.date >= start_date) &
            (df["date"].dt.date <= end_date)
        ]

        # Signal type filter
        signal_filter = st.multiselect(
            "Show",
            options=["symptom", "trigger"],
            default=["symptom", "trigger"],
            key="tl_signal_filter",
        )
        df = df[df["signal_type"].isin(signal_filter)]

    # Charts
    _render_event_scatter(df, patterns)
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    _render_symptom_heatmap(df)
    _render_lag_distribution(patterns)

    # Raw events table (collapsed)
    if not df.empty:
        with st.expander(f"📋 Raw Events ({len(df)} rows)", expanded=False):
            display_df = df[["date", "signal", "signal_type", "severity", "session_id", "text"]].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "date": "Date",
                    "signal": "Signal",
                    "signal_type": "Type",
                    "severity": "Severity",
                    "session_id": "Session",
                    "text": "Excerpt",
                },
                hide_index=True,
            )


__all__ = ["render"]