"""
ThreatLens Dashboard Metric Cards & Badges

Provides styled HTML badges and cards for SOC status, model performance metrics,
predictions, and provenance indicators.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st


def render_status_badge(status_text: str, is_healthy: bool = True) -> str:
    """Return HTML string for a colored status badge."""
    css_class = "badge-healthy" if is_healthy else "badge-danger"
    icon = "●" if is_healthy else "✖"
    return f'<span class="{css_class}">{icon} {status_text}</span>'


def render_provenance_badge(provenance_text: str) -> None:
    """Render a styled badge displaying dataset provenance."""
    st.markdown(
        f"""
        <div style="margin: 8px 0 12px 0;">
            <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Dataset Provenance:</span>
            <span class="badge-warning" style="margin-left: 6px; font-family: monospace;">{provenance_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_classification_card(
    prediction_label: str,
    raw_probability: float,
    calibrated_probability: Optional[float],
    threshold: float,
    model_version: str = "1.0.0",
) -> None:
    """
    Render a prominent classification decision card with risk score metrics.

    Args:
        prediction_label: 'BENIGN' or 'MALICIOUS'.
        raw_probability: Uncalibrated probability [0.0, 1.0].
        calibrated_probability: Platt/isotonic calibrated probability or None.
        threshold: Decision threshold applied.
        model_version: Model version string.
    """
    is_malicious = prediction_label.upper() == "MALICIOUS"
    border_color = "#ef4444" if is_malicious else "#10b981"
    bg_tint = "rgba(239, 68, 68, 0.08)" if is_malicious else "rgba(16, 185, 129, 0.08)"
    icon = "☣️ MALICIOUS" if is_malicious else "🛡️ BENIGN"
    badge_class = "badge-danger" if is_malicious else "badge-healthy"

    cal_prob_str = f"{calibrated_probability:.4f}" if calibrated_probability is not None else "N/A"

    st.markdown(
        f"""
        <div class="soc-card" style="border: 2px solid {border_color}; background-color: {bg_tint};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 1.4rem; font-weight: 800; color: {border_color};">{icon}</div>
                <div><span class="{badge_class}">Threshold Applied: {threshold:.2f}</span></div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-top: 8px;">
                <div class="metric-box">
                    <div class="metric-label">Model Score (Calibrated)</div>
                    <div class="metric-value" style="color: {border_color};">{cal_prob_str}</div>
                    <div class="metric-sub">P(Malicious | Calibrated)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Raw Probability</div>
                    <div class="metric-value" style="color: #94a3b8;">{raw_probability:.4f}</div>
                    <div class="metric-sub">LightGBM predict_proba</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Decision Boundary</div>
                    <div class="metric-value" style="color: #38bdf8;">{threshold:.2f}</div>
                    <div class="metric-sub">Operational Cutoff</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Model Release</div>
                    <div class="metric-value" style="color: #cbd5e1; font-size: 1.2rem;">v{model_version}</div>
                    <div class="metric-sub">LightGBM EMBER</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: Dict[str, Any]) -> None:
    """
    Render a responsive row of SOC model performance cards.

    Args:
        metrics: Dictionary containing accuracy, precision, recall, f1, fpr, fnr, etc.
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        acc = metrics.get("accuracy", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{acc:.1%}</div>
                <div class="metric-sub">Overall Test Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        prec = metrics.get("precision", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{prec:.1%}</div>
                <div class="metric-sub">True Positive Purity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        rec = metrics.get("recall", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Recall / TPR</div>
                <div class="metric-value">{rec:.1%}</div>
                <div class="metric-sub">Malware Caught</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        fnr = metrics.get("false_negative_rate", 0.0)
        st.markdown(
            f"""
            <div class="metric-box" style="border-color: rgba(239, 68, 68, 0.4);">
                <div class="metric-label" style="color: #f87171;">FNR (Missed)</div>
                <div class="metric-value" style="color: #f87171;">{fnr:.1%}</div>
                <div class="metric-sub">Critical Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        fpr = metrics.get("false_positive_rate", 0.0)
        st.markdown(
            f"""
            <div class="metric-box" style="border-color: rgba(245, 158, 11, 0.4);">
                <div class="metric-label" style="color: #fbbf24;">FPR (Fatigue)</div>
                <div class="metric-value" style="color: #fbbf24;">{fpr:.1%}</div>
                <div class="metric-sub">False Alarm Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
