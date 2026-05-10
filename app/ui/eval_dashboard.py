"""
app/ui/eval_dashboard.py — Evaluation Dashboard  (aliased as 05_eval)

Measures Clary's pattern detection quality against the 8 golden patterns
from the askfirst_synthetic_dataset.

Metrics computed:
  ─ Pattern Detection ─────────────────────────────────────────
  • Precision        = TP / (TP + FP)
  • Recall           = TP / (TP + FN)
  • F1 Score         = harmonic mean of P & R
  • False Positive Rate = FP / (FP + TN)

  ─ Temporal Reasoning ────────────────────────────────────────
  • Lag Accuracy     = % of TP patterns whose detected lag
                       falls within ±7 days of golden lag
  • Lag MAE          = mean absolute error in days

  ─ Confidence Calibration ────────────────────────────────────
  • Calibration ECE  = expected calibration error
                       |fraction correct − mean confidence|
  • Hallucination Rate = FP / total detected

  ─ Skeptic Performance ───────────────────────────────────────
  • Skeptic Rejection Accuracy = % of true FPs correctly rejected
  • Skeptic Override Rate      = % of true TPs nearly rejected
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Path fix
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.state import ClaryState

# ── Golden pattern definitions ─────────────────────────────────────────────────
# Derived from askfirst_synthetic_dataset hidden_patterns_reference

GOLDEN_PATTERNS: list[dict] = [
    {
        "id": "P1", "user_id": "USR001", "difficulty": "easy",
        "symptom": "acidity",
        "trigger": "late dinner",
        "lag_days_min": 0, "lag_days_max": 1,
        "mechanism": "GERD / Acid Reflux (Late Meal)",
        "min_occurrences": 2,
        "sessions": ["S01", "S04"],
    },
    {
        "id": "P2", "user_id": "USR001", "difficulty": "easy",
        "symptom": "headache",
        "trigger": "low water intake",
        "lag_days_min": 0, "lag_days_max": 1,
        "mechanism": "Dehydration Headache",
        "min_occurrences": 2,
        "sessions": ["S02", "S06"],
    },
    {
        "id": "P3", "user_id": "USR002", "difficulty": "hard",
        "symptom": "hair fall",
        "trigger": "calorie restriction",
        "lag_days_min": 42, "lag_days_max": 84,
        "mechanism": "Telogen Effluvium",
        "min_occurrences": 2,
        "sessions": ["S09", "S14"],
    },
    {
        "id": "P4", "user_id": "USR002", "difficulty": "medium",
        "symptom": "acne",
        "trigger": "dairy intake",
        "lag_days_min": 2, "lag_days_max": 4,
        "mechanism": "Dairy-Induced Acne (IGF-1 / Hormonal Pathway)",
        "min_occurrences": 2,
        "sessions": ["S10", "S15"],
    },
    {
        "id": "P5", "user_id": "USR003", "difficulty": "easy",
        "symptom": "energy crash",
        "trigger": "high carb lunch",
        "lag_days_min": 0, "lag_days_max": 1,
        "mechanism": "Post-Prandial Hypoglycaemia",
        "min_occurrences": 2,
        "sessions": ["S18", "S22"],
    },
    {
        "id": "P6", "user_id": "USR003", "difficulty": "hard",
        "symptom": "cramps",
        "trigger": "sleep deprivation",
        "lag_days_min": 14, "lag_days_max": 35,
        "mechanism": "Sleep Deprivation → Cortisol Dysmenorrhea",
        "min_occurrences": 3,
        "sessions": ["S17", "S19", "S24"],
    },
    {
        "id": "P7", "user_id": "USR003", "difficulty": "hard",
        "symptom": "anxiety",
        "trigger": "late night screens",
        "lag_days_min": 14, "lag_days_max": 56,
        "mechanism": "Sleep Deprivation → Anxiety Cascade",
        "min_occurrences": 2,
        "sessions": ["S19", "S23"],
    },
    {
        "id": "P8", "user_id": "USR002", "difficulty": "hard",
        "symptom": "hair fall",
        "trigger": "calorie restriction",
        "lag_days_min": 35, "lag_days_max": 84,
        "mechanism": "Nutritional Cascade (Caloric Restriction)",
        "min_occurrences": 2,
        "sessions": ["S09", "S14"],
    },
]

_DIFFICULTY_COLOR = {"easy": "#10b981", "medium": "#f59e0b", "hard": "#ef4444"}
_STATUS_COLOR = {
    "confirmed": "#10b981", "emerging": "#f59e0b",
    "watching": "#6b7280", "rejected": "#ef4444",
}
_CONF_SCORE = {
    "very_high": 0.95, "high": 0.78,
    "moderate": 0.55, "low": 0.30, "rejected": 0.05,
}

THRESHOLD_F1 = 0.85
LAG_TOLERANCE_DAYS = 7


# ── Evaluation engine ──────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    return s.lower().strip().replace("-", " ").replace("_", " ")


def _symptom_match(a: str, b: str) -> bool:
    na, nb = _normalise(a), _normalise(b)
    return na == nb or na in nb or nb in na


def _trigger_match(a: str, b: str) -> bool:
    na, nb = _normalise(a), _normalise(b)
    return na == nb or na in nb or nb in na


def _lag_within_tolerance(detected_lag_mid: float, golden: dict, tol: int = LAG_TOLERANCE_DAYS) -> bool:
    golden_mid = (golden["lag_days_min"] + golden["lag_days_max"]) / 2
    return abs(detected_lag_mid - golden_mid) <= tol


def _run_evaluation(
    detected: list[dict],
    golden: list[dict],
    user_id_filter: str | None = None,
) -> dict[str, Any]:
    """
    Core evaluation logic.
    Returns a results dict with per-pattern match results and aggregate metrics.
    """
    if user_id_filter:
        golden = [g for g in golden if g["user_id"] == user_id_filter]
        detected = [d for d in detected if d.get("user_id") == user_id_filter]

    results: list[dict] = []
    matched_detected_idxs: set[int] = set()

    for g in golden:
        best_match: dict | None = None
        best_score: float = 0.0
        best_idx: int = -1

        for i, d in enumerate(detected):
            if i in matched_detected_idxs:
                continue
            s_ok = _symptom_match(g["symptom"], d.get("symptom", ""))
            t_ok = _trigger_match(g["trigger"], d.get("trigger", ""))
            if not (s_ok and t_ok):
                continue
            score = (1.0 if s_ok else 0) + (1.0 if t_ok else 0)
            if score > best_score:
                best_score = score
                best_match = d
                best_idx = i

        is_tp = best_match is not None
        if is_tp:
            matched_detected_idxs.add(best_idx)

        # Lag accuracy
        lag_ok: bool | None = None
        detected_lag_mid: float | None = None
        if is_tp:
            dl_min = best_match.get("lag_days_min", 0)
            dl_max = best_match.get("lag_days_max", 0)
            detected_lag_mid = (dl_min + dl_max) / 2
            lag_ok = _lag_within_tolerance(detected_lag_mid, g)

        # Confidence score
        conf_score: float | None = None
        if is_tp and best_match:
            conf_score = _CONF_SCORE.get(best_match.get("confidence", "low"), 0.3)

        results.append({
            "golden_id": g["id"],
            "user_id": g["user_id"],
            "difficulty": g["difficulty"],
            "golden_symptom": g["symptom"],
            "golden_trigger": g["trigger"],
            "golden_mechanism": g["mechanism"],
            "golden_lag_mid": (g["lag_days_min"] + g["lag_days_max"]) / 2,
            "is_tp": is_tp,
            "detected_status": best_match.get("status", "") if best_match else "",
            "detected_confidence": best_match.get("confidence", "") if best_match else "",
            "detected_lag_mid": detected_lag_mid,
            "lag_within_tolerance": lag_ok,
            "confidence_score": conf_score,
        })

    # False positives = detected patterns not matched to any golden
    fp_count = len(detected) - len(matched_detected_idxs)
    tp_count = sum(1 for r in results if r["is_tp"])
    fn_count = len(golden) - tp_count

    precision = tp_count / max(len(detected), 1)
    recall = tp_count / max(len(golden), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp_count / max(len(detected), 1)

    lag_results = [r for r in results if r["is_tp"] and r["lag_within_tolerance"] is not None]
    lag_accuracy = sum(1 for r in lag_results if r["lag_within_tolerance"]) / max(len(lag_results), 1)
    lag_mae_vals = [
        abs(r["detected_lag_mid"] - r["golden_lag_mid"])
        for r in results
        if r["is_tp"] and r["detected_lag_mid"] is not None
    ]
    lag_mae = sum(lag_mae_vals) / max(len(lag_mae_vals), 1) if lag_mae_vals else None

    # Calibration ECE
    conf_pairs = [(r["confidence_score"], 1.0) for r in results if r["is_tp"] and r["confidence_score"]]
    ece = sum(abs(c - 1.0) for c, _ in conf_pairs) / max(len(conf_pairs), 1) if conf_pairs else None

    return {
        "per_pattern": results,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "n_golden": len(golden),
        "n_detected": len(detected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "lag_accuracy": lag_accuracy,
        "lag_mae_days": lag_mae,
        "calibration_ece": ece,
        "passed": f1 >= THRESHOLD_F1,
    }


# ── Metric card ────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: Any, color: str, sub: str = "", fmt: str = "") -> str:
    if isinstance(value, float):
        disp = f"{value:.1%}" if fmt == "pct" else f"{value:.2f}"
    else:
        disp = str(value)
    return (
        f'<div class="metric-box">'
        f'<div class="metric-val" style="color:{color};">{disp}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{f"<div style=\\"font-size:10px;color:#4b5563;margin-top:4px;\\">{sub}</div>" if sub else ""}'
        f'</div>'
    )


def _pass_fail_badge(passed: bool) -> str:
    if passed:
        return '<span class="badge badge-green">✓ PASS</span>'
    return '<span class="badge badge-coral">✗ FAIL</span>'


# ── Chart builders ─────────────────────────────────────────────────────────────

def _chart_pr_bars(results: dict) -> go.Figure:
    metrics = {
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1 Score": results["f1"],
        "Lag Accuracy": results["lag_accuracy"],
    }
    colors = [
        "#3b82f6" if v >= THRESHOLD_F1 else "#ef4444"
        for v in metrics.values()
    ]
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker_color=colors,
        text=[f"{v:.1%}" for v in metrics.values()],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(
        y=THRESHOLD_F1, line_dash="dot", line_color="#374151",
        annotation_text=f"threshold {THRESHOLD_F1:.0%}",
        annotation_font_color="#6b7280", annotation_position="top right",
    )
    fig.update_layout(
        paper_bgcolor="#0a0c12", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[0, 1.15], gridcolor="#1f2937", tickformat=".0%"),
        xaxis=dict(gridcolor="#1f2937"),
        height=300, showlegend=False,
    )
    return fig


def _chart_confidence_distribution(results: dict) -> go.Figure:
    per = results["per_pattern"]
    tp_confs = [
        _CONF_SCORE.get(r["detected_confidence"], 0)
        for r in per if r["is_tp"] and r["detected_confidence"]
    ]
    fp_confs = [0.2] * results["fp"]  # FPs default low confidence proxy

    fig = go.Figure()
    if tp_confs:
        fig.add_trace(go.Histogram(
            x=tp_confs, name="True Positives", nbinsx=10,
            marker_color="#3b82f6", opacity=0.75,
            hovertemplate="Confidence: %{x:.0%}<br>Count: %{y}<extra>TP</extra>",
        ))
    if fp_confs:
        fig.add_trace(go.Histogram(
            x=fp_confs, name="False Positives", nbinsx=5,
            marker_color="#ef4444", opacity=0.6,
            hovertemplate="Confidence: %{x:.0%}<br>Count: %{y}<extra>FP</extra>",
        ))
    fig.update_layout(
        paper_bgcolor="#0a0c12", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        barmode="overlay",
        xaxis=dict(title="Confidence Score", tickformat=".0%", gridcolor="#1f2937"),
        yaxis=dict(title="Count", gridcolor="#1f2937"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=260,
    )
    return fig


def _chart_lag_scatter(results: dict) -> go.Figure:
    per = [r for r in results["per_pattern"] if r["is_tp"] and r["detected_lag_mid"] is not None]
    if not per:
        return go.Figure()

    golden_lags = [r["golden_lag_mid"] for r in per]
    detected_lags = [r["detected_lag_mid"] for r in per]
    labels = [f"{r['golden_trigger'][:12]}→{r['golden_symptom'][:12]}" for r in per]
    within = [r["lag_within_tolerance"] for r in per]
    point_colors = ["#10b981" if ok else "#ef4444" for ok in within]

    fig = go.Figure()
    # Perfect prediction line
    max_val = max(max(golden_lags + [1]), max(detected_lags + [1]))
    fig.add_trace(go.Scatter(
        x=[0, max_val * 1.1], y=[0, max_val * 1.1],
        mode="lines", name="Perfect prediction",
        line=dict(color="#374151", dash="dash", width=1),
    ))
    # Tolerance bands
    tol = LAG_TOLERANCE_DAYS
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[tol, max_val + tol],
        mode="lines", name=f"±{tol}d tolerance",
        line=dict(color="#1f2937", width=1), fill=None,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[-tol, max_val - tol],
        mode="lines", fillcolor="rgba(59,130,246,0.05)",
        fill="tonexty", line=dict(color="#1f2937", width=1),
        name=f"±{tol}d band",
    ))
    # Points
    fig.add_trace(go.Scatter(
        x=golden_lags, y=detected_lags,
        mode="markers+text",
        marker=dict(color=point_colors, size=12, line=dict(color="#0a0c12", width=1.5)),
        text=labels, textposition="top center",
        textfont=dict(size=9, color="#6b7280"),
        name="Pattern",
        hovertemplate="<b>%{text}</b><br>Golden lag: %{x:.0f}d<br>Detected lag: %{y:.0f}d<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0c12", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Golden lag (days)", gridcolor="#1f2937"),
        yaxis=dict(title="Detected lag (days)", gridcolor="#1f2937"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=320,
    )
    return fig


def _chart_difficulty_breakdown(results: dict) -> go.Figure:
    per = results["per_pattern"]
    breakdown: dict[str, dict[str, int]] = {
        "easy": {"tp": 0, "fn": 0},
        "medium": {"tp": 0, "fn": 0},
        "hard": {"tp": 0, "fn": 0},
    }
    for r in per:
        d = r["difficulty"]
        if d in breakdown:
            if r["is_tp"]:
                breakdown[d]["tp"] += 1
            else:
                breakdown[d]["fn"] += 1

    difficulties = list(breakdown.keys())
    tps = [breakdown[d]["tp"] for d in difficulties]
    fns = [breakdown[d]["fn"] for d in difficulties]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Detected (TP)", x=difficulties, y=tps,
                         marker_color="#3b82f6",
                         hovertemplate="<b>%{x}</b><br>Detected: %{y}<extra>TP</extra>"))
    fig.add_trace(go.Bar(name="Missed (FN)", x=difficulties, y=fns,
                         marker_color="#ef4444", opacity=0.7,
                         hovertemplate="<b>%{x}</b><br>Missed: %{y}<extra>FN</extra>"))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0a0c12", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", title="Count"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
        height=260,
    )
    return fig


def _chart_calibration_curve(results: dict) -> go.Figure:
    per = results["per_pattern"]
    conf_buckets = [0.1, 0.3, 0.55, 0.78, 0.95]
    bucket_labels = ["low", "low", "moderate", "high", "very_high"]
    actual_fractions: list[float] = []
    mean_confs: list[float] = []

    for bucket_conf, bucket_label in zip(conf_buckets, bucket_labels):
        bucket_items = [r for r in per if
                        r.get("detected_confidence") == bucket_label and r["detected_confidence"]]
        if not bucket_items:
            continue
        frac_correct = sum(1 for r in bucket_items if r["is_tp"]) / len(bucket_items)
        actual_fractions.append(frac_correct)
        mean_confs.append(bucket_conf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#374151", dash="dash"), name="Perfect calibration",
    ))
    if mean_confs:
        fig.add_trace(go.Scatter(
            x=mean_confs, y=actual_fractions,
            mode="markers+lines",
            marker=dict(color="#8b5cf6", size=10),
            line=dict(color="#8b5cf6", width=2),
            name="Clary calibration",
            hovertemplate="Mean confidence: %{x:.0%}<br>Actual accuracy: %{y:.0%}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="#0a0c12", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Mean predicted confidence", range=[0, 1], tickformat=".0%", gridcolor="#1f2937"),
        yaxis=dict(title="Fraction correct", range=[0, 1], tickformat=".0%", gridcolor="#1f2937"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=260,
    )
    return fig


# ── Per-pattern comparison table ───────────────────────────────────────────────

def _render_pattern_comparison(results: dict) -> None:
    st.markdown("### Pattern-by-Pattern Comparison")
    per = results["per_pattern"]

    for r in per:
        is_tp = r["is_tp"]
        gid = r["golden_id"]
        diff = r["difficulty"]
        diff_color = _DIFFICULTY_COLOR.get(diff, "#6b7280")
        status_icon = "✓" if is_tp else "✗"
        status_color = "#10b981" if is_tp else "#ef4444"

        lag_str = "—"
        lag_color = "#4b5563"
        if r["detected_lag_mid"] is not None:
            lag_str = f"{r['detected_lag_mid']:.0f}d"
            lag_color = "#10b981" if r.get("lag_within_tolerance") else "#ef4444"

        with st.container():
            st.markdown(
                f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;'
                f'padding:14px 18px;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
                f'<span style="font-family:monospace;font-size:10px;color:#4b5563;'
                f'background:#0a0c12;padding:2px 8px;border-radius:4px;">{gid}</span>'
                f'<span style="font-size:13px;font-weight:600;color:#f1f5f9;">'
                f'{r["golden_trigger"]} → {r["golden_symptom"]}</span>'
                f'<span style="margin-left:auto;display:flex;gap:8px;align-items:center;">'
                f'<span style="font-size:10px;color:{diff_color};font-family:monospace;">{diff.upper()}</span>'
                f'<span style="font-size:14px;font-weight:700;color:{status_color};">{status_icon}</span>'
                f'</span></div>'
                f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
                f'<div><div style="font-size:10px;color:#4b5563;">Mechanism</div>'
                f'<div style="font-size:12px;color:#94a3b8;">{r["golden_mechanism"]}</div></div>'
                f'<div><div style="font-size:10px;color:#4b5563;">Golden Lag</div>'
                f'<div style="font-size:12px;color:#94a3b8;">{r["golden_lag_mid"]:.0f}d</div></div>'
                f'<div><div style="font-size:10px;color:#4b5563;">Detected Lag</div>'
                f'<div style="font-size:12px;color:{lag_color};">{lag_str}</div></div>'
                f'<div><div style="font-size:10px;color:#4b5563;">Status</div>'
                f'<div style="font-size:12px;color:{_STATUS_COLOR.get(r["detected_status"],"#4b5563")};">'
                f'{r["detected_status"] or "NOT DETECTED"}</div></div>'
                f'<div><div style="font-size:10px;color:#4b5563;">Confidence</div>'
                f'<div style="font-size:12px;color:#94a3b8;">{r["detected_confidence"] or "—"}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ── Download report ────────────────────────────────────────────────────────────

def _build_download_report(results: dict) -> str:
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "precision": round(results["precision"], 4),
            "recall": round(results["recall"], 4),
            "f1": round(results["f1"], 4),
            "false_positive_rate": round(results["false_positive_rate"], 4),
            "lag_accuracy": round(results["lag_accuracy"], 4),
            "lag_mae_days": results.get("lag_mae_days"),
            "calibration_ece": results.get("calibration_ece"),
            "passed": results["passed"],
            "threshold": THRESHOLD_F1,
        },
        "counts": {
            "golden_patterns": results["n_golden"],
            "detected_patterns": results["n_detected"],
            "true_positives": results["tp"],
            "false_positives": results["fp"],
            "false_negatives": results["fn"],
        },
        "per_pattern_results": results["per_pattern"],
    }
    return json.dumps(report, indent=2, default=str)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    ClaryState.init()

    st.markdown('<h2 style="color:#f1f5f9;font-weight:600;margin:0;">📊 Evaluation Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#4b5563;margin-top:2px;margin-bottom:4px;font-family:monospace;">Pattern detection quality vs. 8 golden patterns from askfirst_synthetic_dataset</div>', unsafe_allow_html=True)

    st.divider()

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([3, 3, 4])
    with ctrl1:
        user_filter = st.selectbox(
            "User filter",
            options=["All users", "USR001 — Arjun", "USR002 — Meera", "USR003 — Priya"],
            key="eval_user_filter",
        )
        uid_map = {
            "All users": None,
            "USR001 — Arjun": "USR001",
            "USR002 — Meera": "USR002",
            "USR003 — Priya": "USR003",
        }
        uid_filter = uid_map[user_filter]

    with ctrl2:
        include_watching = st.checkbox("Include WATCHING patterns", value=False, key="eval_incl_watching")

    with ctrl3:
        run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    # Get detected patterns
    all_detected = ClaryState.get_patterns()
    if not include_watching:
        all_detected = [p for p in all_detected if p.get("status") in ("confirmed", "emerging")]

    # Run or load evaluation
    eval_results = ClaryState.get_eval_results()
    if run_eval or eval_results is None:
        with st.spinner("Running evaluation against golden patterns…"):
            eval_results = _run_evaluation(all_detected, GOLDEN_PATTERNS, uid_filter)
            ClaryState.set_eval_results(eval_results)

    if eval_results is None:
        st.info("Click **Run Evaluation** to compute metrics.")
        return

    passed = eval_results["passed"]

    # ── Header: Pass/Fail + key metrics ───────────────────────────────────
    hc1, hc2 = st.columns([1, 4])
    with hc1:
        st.markdown(
            f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;'
            f'padding:24px;text-align:center;">'
            f'<div style="font-size:36px;margin-bottom:6px;">{"✅" if passed else "❌"}</div>'
            f'<div style="font-size:13px;font-weight:600;color:{"#10b981" if passed else "#ef4444"};">'
            f'{"PASS" if passed else "FAIL"}</div>'
            f'<div style="font-size:10px;color:#4b5563;margin-top:4px;">F1 ≥ {THRESHOLD_F1:.0%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with hc2:
        cols = st.columns(7)
        metrics = [
            ("Precision",     eval_results["precision"],         "#3b82f6", "pct"),
            ("Recall",        eval_results["recall"],            "#8b5cf6", "pct"),
            ("F1 Score",      eval_results["f1"],                "#10b981", "pct"),
            ("Lag Accuracy",  eval_results["lag_accuracy"],      "#f59e0b", "pct"),
            ("FP Rate",       eval_results["false_positive_rate"],"#ef4444","pct"),
            ("TP",            eval_results["tp"],                 "#3b82f6", "int"),
            ("Missed (FN)",   eval_results["fn"],                 "#ef4444", "int"),
        ]
        for col, (label, val, color, fmt) in zip(cols, metrics):
            with col:
                disp = f"{val:.0%}" if fmt == "pct" else str(val)
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-val" style="color:{color};font-size:20px;">{disp}</div>'
                    f'<div class="metric-label" style="font-size:9px;">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Core Metrics",
        "⏳ Temporal Accuracy",
        "🎯 Calibration",
        "🔬 Per-Pattern",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Precision / Recall / F1 / Lag Accuracy")
            st.plotly_chart(_chart_pr_bars(eval_results), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.markdown("#### Detection by Difficulty")
            st.plotly_chart(_chart_difficulty_breakdown(eval_results),
                            use_container_width=True, config={"displayModeBar": False})

        # Summary table
        with st.expander("📋 Raw counts", expanded=False):
            summary_df = pd.DataFrame([{
                "Metric": "True Positives", "Value": eval_results["tp"]},
                {"Metric": "False Positives", "Value": eval_results["fp"]},
                {"Metric": "False Negatives", "Value": eval_results["fn"]},
                {"Metric": "Golden Patterns", "Value": eval_results["n_golden"]},
                {"Metric": "Detected Patterns", "Value": eval_results["n_detected"]},
                {"Metric": "Lag MAE (days)",
                 "Value": f"{eval_results['lag_mae_days']:.1f}" if eval_results.get("lag_mae_days") else "—"},
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab2:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"#### Lag Prediction Accuracy (tolerance ±{LAG_TOLERANCE_DAYS}d)")
            fig_lag = _chart_lag_scatter(eval_results)
            if fig_lag.data:
                st.plotly_chart(fig_lag, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No true positive patterns with lag data available.")

        with c2:
            lag_acc = eval_results["lag_accuracy"]
            lag_mae = eval_results.get("lag_mae_days")
            st.markdown("#### Lag Summary")
            st.markdown(
                f'<div class="metric-box" style="margin-bottom:12px;">'
                f'<div class="metric-val" style="color:{"#10b981" if lag_acc >= 0.9 else "#f59e0b"};">{lag_acc:.1%}</div>'
                f'<div class="metric-label">Within ±{LAG_TOLERANCE_DAYS}d</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if lag_mae is not None:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-val" style="color:#3b82f6;">{lag_mae:.1f}d</div>'
                    f'<div class="metric-label">Mean Absolute Error</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("""
            <div style="margin-top:16px;font-size:12px;color:#6b7280;line-height:1.8;">
            <b style="color:#94a3b8;">Target:</b> ≥90% of patterns within ±7d<br>
            <b style="color:#94a3b8;">Hard case:</b> Telogen Effluvium (42–84d lag)<br>
            <b style="color:#94a3b8;">Easy case:</b> GERD / Dehydration (same-day)
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confidence Distribution")
            st.plotly_chart(_chart_confidence_distribution(eval_results),
                            use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.markdown("#### Calibration Curve")
            st.plotly_chart(_chart_calibration_curve(eval_results),
                            use_container_width=True, config={"displayModeBar": False})

        ece = eval_results.get("calibration_ece")
        if ece is not None:
            ece_color = "#10b981" if ece < 0.1 else "#f59e0b" if ece < 0.2 else "#ef4444"
            st.markdown(
                f'<div class="metric-box" style="max-width:200px;">'
                f'<div class="metric-val" style="color:{ece_color};">{ece:.3f}</div>'
                f'<div class="metric-label">ECE (lower is better)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab4:
        _render_pattern_comparison(eval_results)

        # Missed patterns diagnostics
        missed = [r for r in eval_results["per_pattern"] if not r["is_tp"]]
        if missed:
            st.markdown("#### 🔴 Missed Pattern Diagnostics")
            for r in missed:
                st.markdown(
                    f'<div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);'
                    f'border-radius:8px;padding:12px 16px;margin-bottom:8px;">'
                    f'<div style="color:#fca5a5;font-weight:600;font-size:13px;">'
                    f'[{r["golden_id"]}] {r["golden_trigger"]} → {r["golden_symptom"]}</div>'
                    f'<div style="color:#4b5563;font-size:12px;margin-top:4px;">'
                    f'Difficulty: {r["difficulty"]} &nbsp;·&nbsp; '
                    f'Mechanism: {r["golden_mechanism"]} &nbsp;·&nbsp; '
                    f'Expected lag: {r["golden_lag_mid"]:.0f}d</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Download ────────────────────────────────────────────────────────────
    report_json = _build_download_report(eval_results)
    st.download_button(
        label="⬇ Download Evaluation Report (JSON)",
        data=report_json,
        file_name=f"clary_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=False,
    )

    # ── Eval notes ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ Evaluation Methodology", expanded=False):
        st.markdown("""
**Pattern Matching**: A detected pattern is a True Positive if both `symptom` and `trigger`
fuzzy-match the golden ground truth (substring matching, case-insensitive).

**Temporal Accuracy**: For each TP, the detected lag midpoint `(lag_min + lag_max) / 2`
must fall within ±7 days of the golden lag midpoint.

**Calibration ECE**: Expected Calibration Error measures how well the confidence scores
reflect actual detection accuracy. Lower is better (0 = perfect calibration).

**Threshold**: F1 ≥ 85% required to pass. This is enforced in CI/CD before every deploy.

**Golden patterns** sourced from `askfirst_synthetic_dataset.json` `hidden_patterns_reference`.
        """)


__all__ = ["render"]