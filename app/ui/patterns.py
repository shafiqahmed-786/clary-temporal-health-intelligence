"""
app/ui/patterns.py — Pattern Dashboard

Features:
  - Pattern cards with lifecycle status and confidence badges
  - Evidence chain timeline within each card
  - Skeptic verdict details with rubric checks
  - Causal mechanism explainers
  - Intervention tracking (behaviour change tracker)
  - Plotly confidence calibration chart
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from app.state import ClaryState


# ── Helpers ────────────────────────────────────────────────────────────────────

_STATUS_META = {
    "confirmed": {"icon": "✓", "color": "#10b981", "border": "3px solid #10b981", "label": "CONFIRMED"},
    "emerging":  {"icon": "◐", "color": "#f59e0b", "border": "3px solid #f59e0b", "label": "EMERGING"},
    "watching":  {"icon": "○", "color": "#6b7280", "border": "1px solid #374151", "label": "WATCHING"},
    "rejected":  {"icon": "✗", "color": "#ef4444", "border": "1px dashed #ef4444", "label": "REJECTED"},
}

_CONFIDENCE_META = {
    "very_high": {"color": "#10b981", "label": "Very High"},
    "high":      {"color": "#3b82f6", "label": "High"},
    "moderate":  {"color": "#f59e0b", "label": "Moderate"},
    "low":       {"color": "#6b7280", "label": "Low"},
    "rejected":  {"color": "#ef4444", "label": "Rejected"},
}

_RUBRIC_LABELS = {
    "min_evidence_passed":           ("Minimum evidence (N≥2)", True),
    "temporal_ordering_passed":      ("Temporal ordering (trigger before symptom)", True),
    "trigger_fully_consistent":      ("Trigger consistent in ALL occurrences", True),
    "lag_plausible":                 ("Lag window biologically plausible", True),
    "dose_response_found":           ("Dose-response relationship detected", True),
    "removal_test_passed":           ("Symptom improved when trigger removed", True),
    "confounders_present":           ("Confounders present", False),   # negative
    "alternative_equally_plausible": ("Alternative explanation equally plausible", False),
    "high_base_rate":                ("High base rate (common occurrence)", False),
}


def _lag_str(lag_min: int, lag_max: int) -> str:
    if lag_max <= 1:
        return "Same day / night"
    if lag_min == lag_max:
        return f"{lag_min} days"
    return f"{lag_min} – {lag_max} days"


def _render_check_row(key: str, value: bool) -> None:
    label, is_positive = _RUBRIC_LABELS.get(key, (key, True))
    if is_positive:
        icon = "✓" if value else "○"
        color = "#10b981" if value else "#374151"
    else:
        # Negative check: False = good, True = bad
        icon = "✗" if value else "✓"
        color = "#ef4444" if value else "#10b981"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
        f'<span style="color:{color};font-weight:700;font-size:12px;">{icon}</span>'
        f'<span style="font-size:12px;color:#94a3b8;">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_evidence_chain(evidence: list[dict], pattern_symptom: str, pattern_trigger: str) -> None:
    """Render a mini timeline of supporting sessions."""
    if not evidence:
        st.caption("No evidence sessions available.")
        return

    st.markdown(
        '<div style="font-size:11px;color:#6b7280;text-transform:uppercase;'
        'letter-spacing:1px;margin-bottom:8px;">Evidence Chain</div>',
        unsafe_allow_html=True,
    )

    for i, ev in enumerate(evidence[:5]):
        date_str = (ev.get("occurred_at") or "")[:10]
        lag = ev.get("lag_days", 0)
        severity = ev.get("symptom_severity", "unknown")
        co_vars = ev.get("co_occurring_variables", [])
        excerpt = ev.get("raw_excerpt", "")[:100]
        session_id = ev.get("session_id", "?")

        sev_color = {
            "severe": "#ef4444", "moderate": "#f59e0b",
            "mild": "#3b82f6", "none": "#10b981"
        }.get(severity, "#6b7280")

        st.markdown(
            f'<div style="display:flex;gap:12px;padding:8px 0;'
            f'border-bottom:1px solid #1f2937;align-items:flex-start;">'
            f'<div style="font-family:monospace;font-size:10px;color:#3b82f6;'
            f'min-width:80px;padding-top:2px;">#{i+1} · {date_str}</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:12px;color:#d1d5db;">{excerpt or f"Symptom: {pattern_symptom}"}</div>'
            f'<div style="font-size:10px;color:#4b5563;margin-top:3px;">'
            f'Lag: <span style="color:#f59e0b;">{lag:.1f}d</span> &nbsp;·&nbsp; '
            f'Severity: <span style="color:{sev_color};">{severity}</span>'
            f'{(" &nbsp;·&nbsp; Confounders: " + ", ".join(co_vars[:2])) if co_vars else ""}'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )


def _render_mechanism_block(mechanism: dict) -> None:
    if not mechanism:
        return
    name = mechanism.get("mechanism_name", "")
    desc = mechanism.get("description", "")
    lag_min = mechanism.get("lag_min_days", 0)
    lag_max = mechanism.get("lag_max_days", 0)
    quality = mechanism.get("match_quality", 0)

    st.markdown(
        f'<div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);'
        f'border-radius:8px;padding:12px 14px;margin-top:10px;">'
        f'<div style="font-size:11px;color:#3b82f6;font-weight:600;margin-bottom:4px;">'
        f'🔬 Biological Mechanism: {name}'
        f'</div>'
        f'<div style="font-size:12px;color:#94a3b8;line-height:1.6;">{desc[:300]}…</div>'
        f'<div style="font-size:10px;color:#4b5563;margin-top:6px;font-family:monospace;">'
        f'Known lag window: {lag_min}–{lag_max} days &nbsp;·&nbsp; Match quality: {quality:.0%}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_pattern_card(pattern: dict, idx: int) -> None:
    """Full pattern card with all details."""
    status = pattern.get("status", "watching")
    meta = _STATUS_META.get(status, _STATUS_META["watching"])
    conf = pattern.get("confidence", "low")
    conf_meta = _CONFIDENCE_META.get(conf, _CONFIDENCE_META["low"])

    symptom = pattern.get("symptom", "—")
    trigger = pattern.get("trigger", "—")
    n = pattern.get("occurrence_count", 0)
    lag_min = pattern.get("lag_days_min", 0)
    lag_max = pattern.get("lag_days_max", 0)
    title = pattern.get("title", f"{trigger.title()} → {symptom.title()}")
    confounders = pattern.get("confounders", [])
    mechanism = pattern.get("lag_registry_match") or {}
    evidence = pattern.get("evidence", [])
    skeptic = pattern.get("skeptic_verdict") or {}
    checks = skeptic.get("checks") or {}
    alternatives = skeptic.get("alternatives", [])
    dissent = skeptic.get("dissent_note", "")
    first_detected = (pattern.get("first_detected_at") or "")[:10]
    last_confirmed = (pattern.get("last_confirmed_at") or "")[:10]

    with st.container():
        st.markdown(
            f'<div style="border-left:{meta["border"]};background:#111827;'
            f'border-top:1px solid #1f2937;border-right:1px solid #1f2937;'
            f'border-bottom:1px solid #1f2937;border-radius:4px 12px 12px 4px;'
            f'padding:18px 20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )

        # Header row
        hc1, hc2 = st.columns([7, 3])
        with hc1:
            st.markdown(
                f'<div style="font-size:15px;font-weight:600;color:#f1f5f9;">{title}</div>'
                f'<div style="font-size:12px;color:#6b7280;margin-top:3px;">'
                f'<span style="color:#94a3b8;">{trigger}</span> → '
                f'<span style="color:#93c5fd;">{symptom}</span></div>',
                unsafe_allow_html=True,
            )
        with hc2:
            st.markdown(
                f'<div style="text-align:right;">'
                f'<span class="badge" style="background:rgba({_hex_to_rgb(meta["color"])},0.15);'
                f'color:{meta["color"]};">{meta["icon"]} {meta["label"]}</span><br><br>'
                f'<span class="badge" style="background:rgba({_hex_to_rgb(conf_meta["color"])},0.15);'
                f'color:{conf_meta["color"]};">confidence: {conf_meta["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Key stats
        st.markdown(
            f'<div style="display:flex;gap:24px;margin:12px 0;flex-wrap:wrap;">'
            f'<div><div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:0.8px;">Occurrences</div>'
            f'<div style="font-size:18px;font-weight:700;color:#3b82f6;font-family:monospace;">{n}</div></div>'
            f'<div><div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:0.8px;">Lag Window</div>'
            f'<div style="font-size:18px;font-weight:700;color:#f59e0b;font-family:monospace;">{_lag_str(lag_min, lag_max)}</div></div>'
            f'<div><div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:0.8px;">First Detected</div>'
            f'<div style="font-size:14px;font-weight:600;color:#94a3b8;">{first_detected}</div></div>'
            f'<div><div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:0.8px;">Last Confirmed</div>'
            f'<div style="font-size:14px;font-weight:600;color:#94a3b8;">{last_confirmed}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Mechanism
        _render_mechanism_block(mechanism)

        # Confounders warning
        if confounders:
            st.markdown(
                f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);'
                f'border-radius:6px;padding:8px 12px;margin-top:10px;font-size:12px;color:#fcd34d;">'
                f'⚠️ Confounders noted: {", ".join(confounders[:4])}</div>',
                unsafe_allow_html=True,
            )

        # Expandable sections
        tab1, tab2, tab3 = st.tabs(["📋 Evidence", "🧐 Skeptic Review", "🔬 Details"])

        with tab1:
            _render_evidence_chain(evidence, symptom, trigger)

        with tab2:
            if checks:
                st.markdown(
                    '<div style="font-size:11px;color:#6b7280;text-transform:uppercase;'
                    'letter-spacing:1px;margin-bottom:8px;">Rubric Checks</div>',
                    unsafe_allow_html=True,
                )
                for key, val in checks.items():
                    if key in _RUBRIC_LABELS:
                        _render_check_row(key, val)

            if alternatives:
                st.markdown(
                    '<div style="font-size:11px;color:#6b7280;margin-top:12px;margin-bottom:4px;">Alternative explanations considered:</div>',
                    unsafe_allow_html=True,
                )
                for alt in alternatives:
                    st.markdown(f'<div style="font-size:12px;color:#9ca3af;padding:3px 0;">· {alt}</div>', unsafe_allow_html=True)

            if dissent:
                st.markdown(
                    f'<div style="background:rgba(239,68,68,0.07);border-left:2px solid #ef4444;'
                    f'border-radius:0 6px 6px 0;padding:8px 12px;margin-top:10px;'
                    f'font-size:12px;color:#fca5a5;">Skeptic dissent: {dissent}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div style="font-size:12px;color:#4b5563;margin-top:8px;">No significant dissent from Skeptic agent.</div>', unsafe_allow_html=True)

        with tab3:
            reasoning = skeptic.get("reasoning_trace", "")
            if reasoning:
                st.code(reasoning, language=None)
            else:
                st.caption("No detailed reasoning trace available.")

        st.markdown("</div>", unsafe_allow_html=True)


# ── Confidence calibration chart ───────────────────────────────────────────────

def _render_confidence_chart(patterns: list[dict]) -> None:
    if not patterns:
        return

    labels = []
    scores = []
    colors = []

    for p in patterns[:8]:
        conf = p.get("confidence", "low")
        conf_meta = _CONFIDENCE_META.get(conf, _CONFIDENCE_META["low"])
        score = {
            "very_high": 0.95, "high": 0.78,
            "moderate": 0.55, "low": 0.3, "rejected": 0.05
        }.get(conf, 0.3)
        short_label = f"{p.get('trigger','?')[:15]}→{p.get('symptom','?')[:15]}"
        labels.append(short_label)
        scores.append(score)
        colors.append(conf_meta["color"])

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0%}" for s in scores],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>Confidence score: %{x:.0%}<extra></extra>",
    ))

    fig.add_vline(
        x=0.5, line_dash="dot", line_color="#374151",
        annotation_text="threshold", annotation_font_color="#4b5563",
        annotation_position="top right",
    )

    fig.update_layout(
        paper_bgcolor="#0a0c12",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=80, t=20, b=20),
        xaxis=dict(
            range=[0, 1.1], gridcolor="#1f2937",
            tickformat=".0%", linecolor="#374151",
        ),
        yaxis=dict(gridcolor="#1f2937"),
        height=max(200, 40 * len(labels)),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Hex helper ─────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    ClaryState.init()
    user_id = ClaryState.get_user_id()

    st.markdown('<h2 style="color:#f1f5f9;font-weight:600;margin:0;">🔍 Pattern Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:#4b5563;margin-top:2px;margin-bottom:16px;font-family:monospace;">{user_id}</div>', unsafe_allow_html=True)

    patterns = ClaryState.get_patterns()
    all_statuses = ["confirmed", "emerging", "watching", "rejected"]

    # Summary strip
    counts = {s: sum(1 for p in patterns if p.get("status") == s) for s in all_statuses}
    cols = st.columns(4)
    for col, (status, cnt) in zip(cols, counts.items()):
        meta = _STATUS_META[status]
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val" style="color:{meta["color"]};">{cnt}</div>'
                f'<div class="metric-label">{status}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    if not patterns:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#4b5563;">
            <div style="font-size:32px;margin-bottom:12px;">🔍</div>
            <div style="font-size:15px;color:#6b7280;">No patterns detected yet.</div>
            <div style="font-size:13px;color:#4b5563;margin-top:8px;">
                Have a few conversations about your health — Clary needs at least
                2 sessions with the same symptom to begin detecting patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Filter controls
    fcol1, fcol2 = st.columns([3, 7])
    with fcol1:
        status_filter = st.multiselect(
            "Filter by status",
            options=all_statuses,
            default=["confirmed", "emerging"],
            key="pat_status_filter",
        )

    filtered = [p for p in patterns if p.get("status") in status_filter] or patterns

    # Two-column layout for charts
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.markdown("#### Pattern Cards")
        for i, pattern in enumerate(filtered):
            _render_pattern_card(pattern, i)
    with ch2:
        st.markdown("#### Confidence Calibration")
        _render_confidence_chart(filtered)

        # Quick stats
        if filtered:
            st.markdown("#### Pattern Summary")
            for p in filtered[:4]:
                sym = p.get("symptom", "—")
                trig = p.get("trigger", "—")
                n = p.get("occurrence_count", 0)
                mech = (p.get("lag_registry_match") or {}).get("mechanism_name", "")
                st.markdown(
                    f'<div style="padding:8px 0;border-bottom:1px solid #1f2937;">'
                    f'<div style="font-size:12px;color:#d1d5db;">{trig} → {sym}</div>'
                    f'<div style="font-size:10px;color:#4b5563;margin-top:2px;">'
                    f'N={n}{f" · {mech}" if mech else ""}</div></div>',
                    unsafe_allow_html=True,
                )


__all__ = ["render"]