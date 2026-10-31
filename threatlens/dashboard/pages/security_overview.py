"""
ThreatLens Dashboard Page 1 — Security Overview

Landing page displaying active backend readiness, status cards, untouched test split
evaluation metrics from GET /metrics/model, and Plotly metric comparison charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import streamlit as st

from threatlens.dashboard.components.charts import (
    create_confusion_matrix_heatmap,
    create_metrics_bar_chart,
)
from threatlens.dashboard.components.metrics import (
    render_metric_row,
    render_provenance_badge,
    render_status_badge,
)
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)
from threatlens.dashboard.state import get_api_client


def _load_fallback_metrics() -> Dict[str, Any]:
    """Load cached classification report from disk if backend is unreachable."""
    report_path = Path("reports/classification_report.json")
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def render_page_security_overview() -> None:
    """Render the Security Overview dashboard landing page."""
    render_header(
        title="Security Overview",
        subtitle="Operational status, active model specifications, and untouched test split metrics.",
        icon="🛡️",
    )

    render_experimental_warning()

    client = get_api_client()

    # Query backend health and metadata
    health_data = {}
    model_info = {}
    metrics_data = {}
    backend_online = False

    try:
        health_data = client.health()
        backend_online = True
    except Exception:
        backend_online = False

    if backend_online:
        try:
            model_info = client.get_model_info()
        except Exception:
            model_info = {}

        try:
            metrics_resp = client.get_model_metrics()
            metrics_data = metrics_resp.get("metrics", {})
            confusion_matrix = metrics_resp.get("confusion_matrix", {})
            provenance = metrics_resp.get("dataset_type", "synthetic-ember-2018")
        except Exception:
            metrics_data = {}
            confusion_matrix = {}
            provenance = "synthetic-ember-2018"
    else:
        # Fallback to local artifacts for offline inspection
        fallback = _load_fallback_metrics()
        metrics_data = fallback.get("metrics", {})
        confusion_matrix = fallback.get("confusion_matrix", {})
        provenance = "synthetic-ember-2018 (offline artifact)"

    # Status Cards Grid
    st.subheader("System & Model Architecture Status")

    active_model_status = health_data.get("model_status", "SYNTHETIC — RESEARCH ONLY") if backend_online else "MODEL UNAVAILABLE"
    if active_model_status == "GENUINE DATASET — EVALUATION MODEL":
        badge_style = "badge-info"
    elif active_model_status == "MODEL UNAVAILABLE":
        badge_style = "badge-danger"
    else:
        badge_style = "badge-warning"

    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">ACTIVE MODEL STATUS:</span>
            <span class="{badge_style}" style="font-size: 0.85rem; margin-left: 8px; font-weight: 700;">{active_model_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        status_label = "Healthy" if backend_online else "Offline"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Backend Service</div>
                <div style="margin-top:6px;">{render_status_badge(status_label, backend_online)}</div>
                <div class="metric-sub">FastAPI Microservice</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        model_loaded = health_data.get("model_loaded", True)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Model Engine</div>
                <div style="margin-top:6px;">{render_status_badge("Loaded" if model_loaded else "Missing", model_loaded)}</div>
                <div class="metric-sub">LightGBM GBDT</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        version = model_info.get("model_version", "1.0.0")
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Model Release</div>
                <div class="metric-value" style="font-size:1.2rem; color:#38bdf8;">v{version}</div>
                <div class="metric-sub">Booster Serialization</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        dim = model_info.get("feature_dimension", 2351)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Feature Dimension</div>
                <div class="metric-value" style="font-size:1.2rem; color:#a78bfa;">{dim}</div>
                <div class="metric-sub">EMBER 2018 Schema</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        cal_status = health_data.get("calibration_status", "calibrated")
        is_cal = cal_status == "calibrated"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Calibration</div>
                <div style="margin-top:6px;">{render_status_badge(cal_status.capitalize(), is_cal)}</div>
                <div class="metric-sub">Platt Sigmoid Scaling</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Explainability</div>
                <div style="margin-top:6px;">{render_status_badge("Active", True)}</div>
                <div class="metric-sub">SHAP TreeExplainer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Dataset Provenance Section
    render_provenance_badge(provenance)
    render_disclaimer_card(
        title="Dataset Provenance: synthetic-ember-2018",
        text=(
            "The active classifier was trained on synthetic data simulating the 2,351-dimensional EMBER 2018 feature schema. "
            "Because the training distribution consists of synthetic random distributions, test accuracy and ROC-AUC are "
            "approximately at chance level (50%). These metrics reflect architectural and pipeline verification, not real-world malware efficacy."
        ),
    )

    # Model Performance Metrics Row
    st.subheader("Model Performance Metrics (Test Split)")
    if metrics_data:
        render_metric_row(metrics_data)
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            fig_bar = create_metrics_bar_chart(metrics_data)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            if confusion_matrix:
                fig_cm = create_confusion_matrix_heatmap(confusion_matrix)
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix details not available from server.")

        # Additional detail metrics
        with st.expander("Detailed Metric Table & Additional Indicators", expanded=False):
            st.json(metrics_data)
    else:
        st.warning("Model evaluation metrics could not be loaded from FastAPI backend or local reports.")
