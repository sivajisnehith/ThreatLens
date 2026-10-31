"""
ThreatLens Dashboard Page 7 — Model Information & Architecture

Displays complete model specifications, training hyperparameters from GET /model/info,
calibration parameters, explicit synthetic limitations, and future development roadmaps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import streamlit as st

from threatlens.dashboard.components.metrics import render_provenance_badge, render_status_badge
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)
from threatlens.dashboard.state import get_api_client


def _load_offline_metadata() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model and calibration metadata files directly if backend is offline."""
    m_info, c_info = {}, {}
    m_path = Path("models/model_metadata.json")
    if m_path.exists():
        try:
            with open(m_path, "r", encoding="utf-8") as f:
                m_info = json.load(f)
        except Exception:
            pass

    c_path = Path("models/calibration_metadata.json")
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c_info = json.load(f)
        except Exception:
            pass
    return m_info, c_info


def render_page_model_information() -> None:
    """Render the Model Information & Architecture dashboard page."""
    render_header(
        title="Model Information & Architecture",
        subtitle="System architecture, training metadata, calibration details, and production readiness disclaimers.",
        icon="ℹ️",
    )

    render_experimental_warning()

    client = get_api_client()
    model_info = {}
    backend_online = False

    try:
        model_info = client.get_model_info()
        backend_online = True
    except Exception:
        backend_online = False

    if not backend_online:
        offline_m, offline_c = _load_offline_metadata()
        model_info = {
            "algorithm": "LightGBM Gradient Boosted Decision Trees (offline)",
            "feature_dimension": offline_m.get("feature_dim", 2351),
            "feature_schema": offline_m.get("feature_pipeline", "EMBER_2018"),
            "dataset_provenance": offline_m.get("provenance_notes", "Stage 1 EMBER 2018 synthetic schema dataset"),
            "model_version": f"{offline_m.get('threatlens_stage', 3)}.0.0",
            "training_metadata": offline_m,
            "calibration_info": {"method": offline_c.get("method", "sigmoid"), "status": "calibrated", "metadata": offline_c},
            "limitations": [
                "SYNTHETIC DATA: The model was trained on synthetic benchmark data.",
                "CHANCE LEVEL: Discrimination ability on real files is approximately 50%.",
                "PIPELINE ISOLATION: Custom 85-dim PE extractor is strictly incompatible with this 2,351-dim model.",
            ],
        }

    # High-level architecture cards
    st.subheader("Classifier Architecture Specifications")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value" style="font-size: 1.1rem; color: #38bdf8;">LightGBM GBDT</div>
                <div class="metric-sub">Histogram-based Trees</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Input Dimension</div>
                <div class="metric-value" style="font-size: 1.1rem; color: #a78bfa;">{model_info.get('feature_dimension', 2351)}</div>
                <div class="metric-sub">EMBER 2018 Schema</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        cal_method = model_info.get("calibration_info", {}).get("method", "sigmoid")
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Calibration Method</div>
                <div class="metric-value" style="font-size: 1.1rem; color: #10b981;">{str(cal_method).capitalize()}</div>
                <div class="metric-sub">Platt Scaling</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        v = model_info.get("model_version", "1.0.0")
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Model Release</div>
                <div class="metric-value" style="font-size: 1.1rem; color: #f59e0b;">v{v}</div>
                <div class="metric-sub">Serialized Booster</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    m_status = model_info.get("model_status", "SYNTHETIC — RESEARCH ONLY")
    status_cls = "badge-info" if "genuine" in m_status.lower() else "badge-warning"
    st.markdown(
        f"""
        <div style="margin-bottom: 12px; display: flex; gap: 12px; align-items: center;">
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">OPERATIONAL STATUS:</span>
            <span class="{status_cls}" style="font-size: 0.85rem; font-weight: 700;">{m_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_provenance_badge(model_info.get("dataset_provenance", "synthetic-ember-2018"))

    # Critical Limitations Section
    st.subheader("Critical Limitations & Production Disclosures")
    limitations = [
        ("1. Synthetic Data Training", "The currently serialized model was trained on synthetic benchmark noise distributions generated to validate pipeline integrity without downloading the full 4 GB EMBER 2018 corpus."),
        ("2. Chance-Level Discrimination", "Model accuracy (~50.5%) and ROC-AUC (~50.7%) on the test split are approximately at coin-flip level. The classifier has learned synthetic statistical noise rather than genuine malware signatures."),
        ("3. Strict Pipeline Isolation", "The 85-dimensional features extracted by ThreatLens' custom static PE extractor (Mode A) cannot be evaluated by this LightGBM model (Mode B). They are mathematically and structurally incompatible."),
        ("4. Hashing Collisions in EMBER", "EMBER 2018 utilizes MurmurHash for imports (1,248 bins), exports (68 bins), and section names. Individual hashed bins cannot be uniquely mapped back to raw API names."),
        ("5. Prohibition on Production Deployment", "This system must NOT be deployed in live SOC or security quarantine pipelines until the classifier is retrained on authentic, verified malware and benign PE datasets."),
    ]

    for title, desc in limitations:
        render_disclaimer_card(title=title, text=desc)

    st.markdown("---")

    # Future Roadmap
    st.subheader("Future Hardening & Production Roadmap")
    roadmap_items = [
        ("Genuine EMBER Retraining", "Download and process the full EMBER 2018 benchmark corpus (~1 million binaries) to produce production-grade threat discrimination."),
        ("End-to-End EMBER PE Feature Extractor", "Implement native Python extraction of the full 2,351 EMBER features directly from uploaded PE binaries."),
        ("Conformal Prediction", "Equip predictions with finite-sample conformal prediction bounds to guarantee rigorous coverage guarantees at specified significance levels."),
        ("Enterprise API Security", "Deploy API Key authentication (X-API-Key), OAuth2 bearer tokens, and IP-based rate limiting (slowapi/Redis)."),
        ("Model Telemetry & Drift Monitoring", "Implement OpenTelemetry and Prometheus exporters to monitor feature distribution drift and inference latency."),
    ]
    for r_title, r_desc in roadmap_items:
        st.markdown(
            f"""
            <div style="margin-bottom: 10px;">
                <span class="badge-info">Planned</span>
                <strong style="margin-left: 8px; color: #f8fafc;">{r_title}</strong>:
                <span style="color: #94a3b8; font-size: 0.88rem;">{r_desc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Hyperparameters Expander
    training_meta = model_info.get("training_metadata", {})
    if training_meta:
        with st.expander("Training Hyperparameters & Artifact Metadata", expanded=False):
            st.json(training_meta)
