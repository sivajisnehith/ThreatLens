"""
ThreatLens Dashboard Page 6 — Simulated Threshold Analysis

Integrates threshold sweep artifacts (reports/threshold_analysis.json) to simulate
operational SOC postures, trade-offs between alert fatigue (FPR) and missed malware (FNR),
and interactive sensitivity inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from threatlens.dashboard.components.charts import create_threshold_tradeoff_chart
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)


def _load_threshold_data() -> Dict[str, Any]:
    """Load precomputed threshold sweep data from disk."""
    report_path = Path("reports/threshold_analysis.json")
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _find_nearest_threshold_point(points: List[Dict[str, Any]], target_t: float) -> Dict[str, Any]:
    """Find the threshold evaluation point nearest to target_t."""
    if not points:
        return {}
    return min(points, key=lambda p: abs(p.get("threshold", 0.5) - target_t))


def render_page_threshold_analysis() -> None:
    """Render the Simulated Threshold Analysis dashboard page."""
    render_header(
        title="Simulated Threshold Analysis",
        subtitle="Operational trade-off simulation: False Negative Rate (missed threats) vs. False Positive Rate (SOC alert fatigue).",
        icon="⚖️",
    )

    render_experimental_warning(
        custom_text=(
            "SIMULATED THRESHOLD ANALYSIS: Threshold metrics are calculated on synthetic validation data. "
            "These curves illustrate SOC operational posture mechanics and must NOT be used for real-world deployment rules."
        )
    )

    threshold_data = _load_threshold_data()
    if not threshold_data:
        st.error(
            "Threshold analysis report 'reports/threshold_analysis.json' not found. "
            "Run Stage 5 calibration / threshold evaluation to generate this artifact."
        )
        return

    points = threshold_data.get("threshold_points", [])
    if not points:
        st.warning("No threshold points recorded in report.")
        return

    # Interactive Threshold Inspector Slider
    st.subheader("Interactive Operational Posture Inspector")
    current_t = st.slider(
        "Simulated Operational Threshold (t):",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("selected_threshold", 0.50)),
        step=0.05,
        key="page6_threshold_slider",
        help="Slide to simulate how adjusting the decision cutoff alters TPR, FPR, FNR, and Precision.",
    )

    matched_point = _find_nearest_threshold_point(points, current_t)

    # Classify Posture
    if current_t <= 0.45:
        posture_title = "HIGH SECURITY POSTURE (Low FNR / High Catch)"
        posture_badge = "badge-danger"
        posture_desc = (
            "Prioritizes catching every malicious sample. False Negative Rate is low, "
            "but False Positive Rate increases, demanding significant SOC analyst alert triage."
        )
    elif current_t >= 0.55:
        posture_title = "LOW INTERRUPTION POSTURE (Low FPR / Low Alert Fatigue)"
        posture_badge = "badge-warning"
        posture_desc = (
            "Prioritizes business continuity by suppressing false alarms. "
            "Minimizes false positives in automated endpoint blocking, but accepts higher risk of missed malware (high FNR)."
        )
    else:
        posture_title = "BALANCED POSTURE (F1-Optimal Trade-off)"
        posture_badge = "badge-healthy"
        posture_desc = (
            "Default operating trade-off balancing detection sensitivity and alert volume."
        )

    st.markdown(
        f"""
        <div class="soc-card" style="border-left: 4px solid #38bdf8;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 1.1rem;">{posture_title}</div>
                <div><span class="{posture_badge}">t = {current_t:.2f}</span></div>
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 6px;">{posture_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Row for Selected Threshold
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        tpr = matched_point.get("recall", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">TPR / Recall</div>
                <div class="metric-value" style="color: #10b981;">{tpr:.1%}</div>
                <div class="metric-sub">Malware Detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        fnr = matched_point.get("false_negative_rate", 0.0)
        st.markdown(
            f"""
            <div class="metric-box" style="border-color: rgba(239, 68, 68, 0.4);">
                <div class="metric-label" style="color: #f87171;">FNR (Missed)</div>
                <div class="metric-value" style="color: #f87171;">{fnr:.1%}</div>
                <div class="metric-sub">Undetected Malware</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        fpr = matched_point.get("false_positive_rate", 0.0)
        st.markdown(
            f"""
            <div class="metric-box" style="border-color: rgba(245, 158, 11, 0.4);">
                <div class="metric-label" style="color: #fbbf24;">FPR (Fatigue)</div>
                <div class="metric-value" style="color: #fbbf24;">{fpr:.1%}</div>
                <div class="metric-sub">Benign Flagged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        prec = matched_point.get("precision", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color: #38bdf8;">{prec:.1%}</div>
                <div class="metric-sub">Alert Fidelity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        f1 = matched_point.get("f1_score", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: #a78bfa;">{f1:.3f}</div>
                <div class="metric-sub">Harmonic Mean</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Plotly Trade-off Curve
    st.subheader("Threshold Sweep Trade-off Curves")
    fig_tradeoff = create_threshold_tradeoff_chart(points, current_threshold=current_t)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # Full Sweep Table in Expander
    with st.expander("Complete Simulated Threshold Sweep Table (19 points)", expanded=False):
        sweep_rows = []
        for p in points:
            sweep_rows.append({
                "Threshold": round(p["threshold"], 2),
                "TPR (Recall)": f"{p['recall']:.1%}",
                "FNR (Missed)": f"{p['false_negative_rate']:.1%}",
                "FPR (False Alarms)": f"{p['false_positive_rate']:.1%}",
                "Precision": f"{p['precision']:.1%}",
                "F1-Score": round(p["f1_score"], 3),
                "Specificity": f"{p['specificity']:.1%}",
                "Balanced Acc": f"{p['balanced_accuracy']:.1%}",
                "TP / FN": f"{p['true_positives']} / {p['false_negatives']}",
                "FP / TN": f"{p['false_positives']} / {p['true_negatives']}",
            })
        st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True)
