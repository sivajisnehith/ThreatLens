"""
ThreatLens Dashboard Page 3 — Prediction Results

Renders the calibrated malware score gauge, classification outcome badge, and interactive
threshold slider that re-evaluates the decision boundary instantly using cached probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import streamlit as st

from threatlens.dashboard.components.charts import create_gauge_chart
from threatlens.dashboard.components.metrics import render_classification_card
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)
from threatlens.dashboard.state import (
    get_api_client,
    set_active_vector,
    set_prediction,
)


def _load_sample_zero() -> Optional[list[float]]:
    """Helper to load sample 0 from test split if user arrives directly."""
    try:
        from pathlib import Path
        test_path = Path("data/processed/X_test.npy")
        if test_path.exists():
            arr = np.load(test_path, mmap_mode="r")
            return arr[0].tolist()
    except Exception:
        pass
    return None


def render_page_prediction_results() -> None:
    """Render the Prediction Results dashboard page."""
    render_header(
        title="Prediction Results",
        subtitle="Calibrated malware risk score gauge and interactive decision threshold evaluation.",
        icon="📊",
    )

    render_experimental_warning()

    active_vector = st.session_state.get("active_vector")
    source_name = st.session_state.get("active_source_name", "Unknown Source")
    active_pred = st.session_state.get("active_prediction")

    # If no sample is loaded, allow one-click quick demo load
    if active_vector is None:
        st.info("No active EMBER feature vector loaded. Select a benchmark sample below or visit the 'Analyze Sample' page.")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📥 Load Benchmark Test Sample #0 (Benign Demo)", use_container_width=True):
                vec = _load_sample_zero()
                if vec:
                    set_active_vector(vec, "Benchmark X_test Sample #0")
                    st.rerun()
        with col_btn2:
            if st.button("📥 Load Benchmark Test Sample #1 (Malicious Demo)", use_container_width=True):
                try:
                    from pathlib import Path
                    test_path = Path("data/processed/X_test.npy")
                    if test_path.exists():
                        arr = np.load(test_path, mmap_mode="r")
                        set_active_vector(arr[1].tolist(), "Benchmark X_test Sample #1")
                        st.rerun()
                except Exception:
                    pass
        return

    st.markdown(f"**Active Sample:** `{source_name}` (2,351 EMBER features)")

    # Execute inference if not already cached
    client = get_api_client()
    current_threshold = st.session_state.get("selected_threshold", 0.50)

    if active_pred is None:
        with st.spinner("Executing prediction via FastAPI /predict..."):
            try:
                active_pred = client.predict(features=active_vector, threshold=current_threshold)
                set_prediction(active_pred)
            except Exception as exc:
                st.error(f"Inference request failed: {exc}")
                return

    raw_prob = float(active_pred.get("raw_probability", 0.5))
    cal_prob = active_pred.get("calibrated_probability")
    effective_score = float(cal_prob) if cal_prob is not None else raw_prob

    # Interactive Threshold Adjustment (Re-evaluates locally without re-running inference!)
    st.subheader("Interactive Operational Decision Boundary")
    new_threshold = st.slider(
        "Decision Cutoff Threshold (t):",
        min_value=0.05,
        max_value=0.95,
        value=float(current_threshold),
        step=0.01,
        key="results_threshold_slider",
        help="Adjusting this slider re-evaluates the classification locally on the already-received probability without contacting the backend.",
    )

    # Dynamic local label re-calculation
    st.session_state.selected_threshold = new_threshold
    is_malicious = effective_score >= new_threshold
    computed_label = "MALICIOUS" if is_malicious else "BENIGN"

    # Gauge and Classification Display
    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        fig_gauge = create_gauge_chart(
            score=effective_score,
            threshold=new_threshold,
            title="Model Score (Calibrated)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_details:
        render_classification_card(
            prediction_label=computed_label,
            raw_probability=raw_prob,
            calibrated_probability=cal_prob,
            threshold=new_threshold,
            model_version=active_pred.get("model_version", "1.0.0"),
        )

    # Interpretation context
    is_genuine = "GENUINE" in str(st.session_state.get("active_model_status", "")).upper()
    notice = (
        "Note: This score is evaluated on genuine EMBER 2018 static features. Static analysis research only."
        if is_genuine
        else "Note: This score is simulated based on synthetic EMBER features. Do not use for real-world triage."
    )
    render_disclaimer_card(
        title="Risk Score Interpretation",
        text=(
            f"The effective calibrated model score of <b>{effective_score:.4f}</b> was evaluated against cutoff <b>{new_threshold:.2f}</b>. "
            f"Because the score is {'at or above' if is_malicious else 'below'} the operational threshold, the system assigns a <b>{computed_label}</b> designation. "
            f"{notice}"
        ),
    )

    # Navigation shortcut to Page 4
    st.markdown("---")
    col_nav1, col_nav2 = st.columns([1, 1])
    with col_nav1:
        if st.button("🔍 Explain This Prediction with SHAP →", type="primary", use_container_width=True):
            st.session_state.nav_target = "Local SHAP Explainability"
            st.rerun()
    with col_nav2:
        if st.button("🗑️ Clear Active Sample", use_container_width=True):
            from threatlens.dashboard.state import clear_active_vector
            clear_active_vector()
            st.rerun()
