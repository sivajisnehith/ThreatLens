"""
ThreatLens Dashboard Page 4 — Local SHAP Explainability

Integrates FastAPI POST /explain to render local feature attributions,
interactive Plotly diverging horizontal contribution bars, top-K selection,
exact additivity verification, and canonical feature descriptions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from threatlens.dashboard.components.charts import create_shap_horizontal_bar
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)
from threatlens.dashboard.state import (
    get_api_client,
    set_active_vector,
    set_explanation,
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


def render_page_local_shap() -> None:
    """Render the Local SHAP Explainability dashboard page."""
    render_header(
        title="Local SHAP Explainability",
        subtitle="Deconstruct individual model predictions into additive feature attributions using TreeSHAP.",
        icon="🔍",
    )

    render_experimental_warning()

    active_vector = st.session_state.get("active_vector")
    source_name = st.session_state.get("active_source_name", "Unknown Source")
    active_expl = st.session_state.get("active_explanation")

    # If no sample is loaded, allow quick demo load
    if active_vector is None:
        st.info("No active feature vector in session. Select a sample below or navigate to 'Analyze Sample'.")
        if st.button("📥 Load Benchmark Test Sample #0 for SHAP Analysis", type="primary"):
            vec = _load_sample_zero()
            if vec:
                set_active_vector(vec, "Benchmark X_test Sample #0")
                st.rerun()
        return

    st.markdown(f"**Explaining Sample:** `{source_name}` (2,351 EMBER features)")

    client = get_api_client()
    top_k = st.slider(
        "Display Top-K Influential Features:",
        min_value=5,
        max_value=40,
        value=int(st.session_state.get("selected_top_k", 15)),
        step=1,
        key="local_shap_top_k_slider",
        help="Controls how many positive and negative feature attributions are returned by the SHAP engine.",
    )
    st.session_state.selected_top_k = top_k

    # Trigger explanation request if not cached or top_k changed
    if active_expl is None or active_expl.get("_requested_top_k") != top_k:
        with st.spinner("Calculating SHAP attributions via FastAPI /explain..."):
            try:
                threshold = float(st.session_state.get("selected_threshold", 0.50))
                active_expl = client.explain(
                    features=active_vector,
                    top_k=top_k,
                    threshold=threshold,
                )
                active_expl["_requested_top_k"] = top_k
                set_explanation(active_expl)
            except Exception as exc:
                st.error(f"SHAP explanation request failed: {exc}")
                return

    # Extract Explanation Metrics
    prediction_label = active_expl.get("prediction_label", "UNKNOWN")
    raw_score = float(active_expl.get("raw_score", 0.0))
    raw_prob = float(active_expl.get("raw_probability", 0.5))
    cal_prob = active_expl.get("calibrated_probability")
    baseline = float(active_expl.get("baseline", 0.0))
    additivity_error = float(active_expl.get("additivity_error", 0.0))
    top_pos = active_expl.get("top_positive_contributors", [])
    top_neg = active_expl.get("top_negative_contributors", [])

    # Overview Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Prediction</div>
                <div class="metric-value" style="color: {'#ef4444' if prediction_label=='MALICIOUS' else '#10b981'}; font-size: 1.1rem;">{prediction_label}</div>
                <div class="metric-sub">Discrete Label</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        cal_str = f"{cal_prob:.4f}" if cal_prob is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Calibrated Score</div>
                <div class="metric-value" style="color: #38bdf8; font-size: 1.1rem;">{cal_str}</div>
                <div class="metric-sub">P(Malicious)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Raw Margin f(x)</div>
                <div class="metric-value" style="color: #cbd5e1; font-size: 1.1rem;">{raw_score:+.4f}</div>
                <div class="metric-sub">Log-odds Output</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Baseline E[f(x)]</div>
                <div class="metric-value" style="color: #94a3b8; font-size: 1.1rem;">{baseline:+.4f}</div>
                <div class="metric-sub">Training Expected Value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Additivity Error</div>
                <div class="metric-value" style="color: #34d399; font-size: 1.1rem;">{additivity_error:.1e}</div>
                <div class="metric-sub">Efficiency Check</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Plotly Horizontal Attribution Bar Chart
    fig_shap = create_shap_horizontal_bar(top_pos=top_pos, top_neg=top_neg, top_k=top_k)
    st.plotly_chart(fig_shap, use_container_width=True)

    # Contextual Disclaimer Card
    render_disclaimer_card(
        title="Understanding SHAP Attributions",
        text=(
            "SHAP explains the model's internal decision logic, <i>not whether the model's prediction is factually correct</i>. "
            "Since this model was trained on synthetic data for pipeline validation, these attributions reflect patterns in synthetic noise "
            "and are intended strictly for educational and architectural demonstration."
        ),
    )

    # Tabular Breakdown of Top Contributors
    st.subheader("Feature Interpretation Breakdown")
    all_contributors = []
    for item in top_pos:
        all_contributors.append({
            "Feature Name": item.get("feature_name", ""),
            "EMBER Group": item.get("feature_group", "").capitalize(),
            "Observed Value": round(float(item.get("feature_value", 0.0)), 4),
            "SHAP Attribution": round(float(item.get("shap_value", 0.0)), 5),
            "Direction": "Increases Malware Score (Risk +)",
            "Description": item.get("description", ""),
        })
    for item in top_neg:
        all_contributors.append({
            "Feature Name": item.get("feature_name", ""),
            "EMBER Group": item.get("feature_group", "").capitalize(),
            "Observed Value": round(float(item.get("feature_value", 0.0)), 4),
            "SHAP Attribution": round(float(item.get("shap_value", 0.0)), 5),
            "Direction": "Decreases Malware Score (Risk -)",
            "Description": item.get("description", ""),
        })

    # Sort descending by absolute SHAP
    all_contributors.sort(key=lambda r: abs(r["SHAP Attribution"]), reverse=True)
    df_contrib = pd.DataFrame(all_contributors)
    st.dataframe(df_contrib, use_container_width=True)
