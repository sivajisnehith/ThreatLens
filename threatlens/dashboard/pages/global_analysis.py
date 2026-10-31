"""
ThreatLens Dashboard Page 5 — Global Feature Analysis

Displays macro-level model explainability across the evaluation cohort:
- Top 20 global features ranked by mean absolute SHAP
- Aggregated feature attribution across the 8 EMBER schema groups
- Metric distinction comparison: SHAP Attribution vs LightGBM Gain vs LightGBM Split
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import streamlit as st

from threatlens.dashboard.components.charts import (
    create_group_importance_chart,
    create_top_features_chart,
)
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)


def _load_global_shap_data() -> Dict[str, Any]:
    """Load serialized global SHAP importance report from disk."""
    report_path = Path("reports/shap_global_importance.json")
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def render_page_global_analysis() -> None:
    """Render the Global Feature Analysis dashboard page."""
    render_header(
        title="Global Feature Analysis",
        subtitle="Cohort-wide feature attribution, EMBER group importance, and tree metric comparisons.",
        icon="🌐",
    )

    render_experimental_warning(
        custom_text=(
            "Global feature importance reflects a model trained on synthetic data. "
            "It must not be interpreted as genuine malware indicators."
        )
    )

    global_data = _load_global_shap_data()
    if not global_data:
        st.error(
            "Could not locate 'reports/shap_global_importance.json'. "
            "Run 'python scripts/explain.py --mode global --data-path data/processed/X_test.npy' to generate it."
        )
        return

    top_features = global_data.get("top_features_by_shap", [])
    group_data = global_data.get("group_importance", {})
    top_gain = global_data.get("top_features_by_gain", [])
    top_split = global_data.get("top_features_by_split", [])
    metric_distinctions = global_data.get("metric_distinctions", {})

    # Top 20 Global Features Chart
    st.subheader("Top 20 Global Features by Mean Absolute SHAP (E[|phi|])")
    fig_top = create_top_features_chart(top_features)
    st.plotly_chart(fig_top, use_container_width=True)

    render_disclaimer_card(
        title="Global SHAP Metric Note",
        text=(
            "Global feature importance reflects a model trained on synthetic data. "
            "It must not be interpreted as genuine malware indicators."
        ),
    )

    st.markdown("---")

    # Group-Level Importance Across the 8 Schema Groups
    st.subheader("Feature Importance by EMBER Schema Group")
    st.markdown(
        "Attribution aggregated across the 8 canonical EMBER 2018 structural feature partitions "
        "covering all 2,351 input dimensions."
    )
    if group_data:
        fig_group = create_group_importance_chart(group_data)
        st.plotly_chart(fig_group, use_container_width=True)

        # Tabular details for groups
        group_rows = []
        for g_name, g_info in group_data.items():
            group_rows.append({
                "EMBER Group": g_name.capitalize(),
                "Feature Dimension Count": g_info.get("feature_count", 0),
                "Mean |SHAP| Attribution": round(float(g_info.get("mean_abs_shap", 0.0)), 5),
                "Most Influential Feature": g_info.get("top_feature_name", "N/A"),
                "Top Feature SHAP": round(float(g_info.get("top_feature_shap", 0.0)), 5),
            })
        df_groups = pd.DataFrame(group_rows).sort_values("Mean |SHAP| Attribution", ascending=False)
        st.dataframe(df_groups, use_container_width=True)

    st.markdown("---")

    # Metric Distinctions: SHAP vs Gain vs Split
    st.subheader("Metric Comparison: SHAP vs. Gain vs. Split")
    render_disclaimer_card(
        title="Critical Metric Distinctions (Non-Interchangeable)",
        text=metric_distinctions.get(
            "non_interchangeability_warning",
            "SHAP importance, Gain, and Split metrics are mathematically distinct and NOT interchangeable.",
        ),
    )

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.markdown("#### 1. SHAP Importance")
        st.caption("E[|phi_i|] on test samples. Measures marginal log-odds impact.")
        df_shap_summary = pd.DataFrame([
            {"Rank": f["rank"], "Feature": f["feature_name"], "|SHAP|": round(f["mean_abs_shap"], 4)}
            for f in top_features[:10]
        ])
        st.dataframe(df_shap_summary, use_container_width=True)

    with col_metric2:
        st.markdown("#### 2. LightGBM Gain")
        st.caption("Total training logloss reduction across all splits.")
        df_gain_summary = pd.DataFrame([
            {"Rank": f["rank"], "Feature": f["feature_name"], "Gain": round(f["gain_importance"], 1)}
            for f in top_gain[:10]
        ])
        st.dataframe(df_gain_summary, use_container_width=True)

    with col_metric3:
        st.markdown("#### 3. LightGBM Split")
        st.caption("Frequency of selection as decision node.")
        df_split_summary = pd.DataFrame([
            {"Rank": f["rank"], "Feature": f["feature_name"], "Splits": f["split_count"]}
            for f in top_split[:10]
        ])
        st.dataframe(df_split_summary, use_container_width=True)
