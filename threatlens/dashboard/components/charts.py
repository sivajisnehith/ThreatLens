"""
ThreatLens Dashboard Plotly Visualization Helpers

Builds consistent, dark-themed, publication-quality Plotly figures for:
- Calibrated risk score gauges
- Local SHAP waterfall and horizontal attribution bars
- Global feature importance by rank and schema group
- Threshold evaluation curves and confusion matrices
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go


DARK_LAYOUT = dict(
    paper_bgcolor="#0d1322",
    plot_bgcolor="#131b2e",
    font=dict(color="#cbd5e1", family="sans-serif", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
)


def create_gauge_chart(
    score: float,
    threshold: float = 0.5,
    title: str = "Model Score (Calibrated)",
) -> go.Figure:
    """
    Create a risk score gauge visualization with operational threshold markers.

    Args:
        score: Probability score between 0.0 and 1.0.
        threshold: Decision cutoff threshold.
        title: Gauge header title.

    Returns:
        Plotly Figure.
    """
    bounded_score = max(0.0, min(1.0, float(score)))

    # Determine dynamic bar color
    if bounded_score >= threshold:
        bar_color = "#ef4444"  # Malicious
    elif bounded_score >= threshold - 0.10:
        bar_color = "#f59e0b"  # Borderline / Suspicious
    else:
        bar_color = "#10b981"  # Benign

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=bounded_score,
            delta={
                "reference": threshold,
                "increasing": {"color": "#ef4444"},
                "decreasing": {"color": "#10b981"},
                "position": "bottom",
            },
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:#94a3b8'>Threshold Cutoff: {threshold:.2f}</span>", "font": {"size": 16, "color": "#f8fafc"}},
            number={"font": {"size": 42, "color": "#f8fafc"}, "valueformat": ".4f"},
            gauge={
                "axis": {"range": [0.0, 1.0], "tickwidth": 1, "tickcolor": "#64748b", "tickformat": ".2f"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#162036",
                "borderwidth": 1,
                "bordercolor": "#22314d",
                "steps": [
                    {"range": [0.0, 0.40], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [0.40, 0.60], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [0.60, 1.0], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#38bdf8", "width": 4},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#131b2e",
        font=dict(color="#cbd5e1"),
        height=260,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def create_shap_horizontal_bar(
    top_pos: List[Dict[str, Any]],
    top_neg: List[Dict[str, Any]],
    top_k: int = 15,
) -> go.Figure:
    """
    Build a horizontal diverging bar chart of local SHAP attributions.

    Args:
        top_pos: List of positive SHAP feature dicts.
        top_neg: List of negative SHAP feature dicts.
        top_k: Maximum total features to display.

    Returns:
        Plotly Figure.
    """
    # Combine and sort by absolute contribution
    all_features = list(top_pos) + list(top_neg)
    all_features.sort(key=lambda item: abs(item.get("shap_value", 0.0)), reverse=True)
    selected = all_features[:top_k]

    # Invert so highest impact appears at the top
    selected.reverse()

    names = [item.get("feature_name", f"Feature_{i}") for i, item in enumerate(selected)]
    shap_vals = [item.get("shap_value", 0.0) for item in selected]
    feat_vals = [item.get("feature_value", 0.0) for item in selected]
    groups = [item.get("feature_group", "unknown") for item in selected]
    descriptions = [item.get("description", "No description") for item in selected]

    colors = ["#ef4444" if val >= 0 else "#38bdf8" for val in shap_vals]

    hover_texts = [
        f"<b>{name}</b><br>"
        f"Group: {group}<br>"
        f"Feature Value: {fval:.4f}<br>"
        f"SHAP Attribution: {sval:+.5f}<br>"
        f"Impact: {'Increases Malware Score' if sval >= 0 else 'Decreases Malware Score'}<br>"
        f"<i>{desc[:80]}...</i>"
        for name, group, fval, sval, desc in zip(names, groups, feat_vals, shap_vals, descriptions)
    ]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=shap_vals,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0d1322", width=1)),
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>Local SHAP Feature Attributions (Top {len(selected)})</b><br>"
                 "<span style='font-size:0.8em;color:#94a3b8'>"
                 "Red: Increases Malware Score | Blue: Decreases Malware Score</span>",
            font=dict(size=14, color="#f8fafc"),
        ),
        xaxis=dict(
            title="SHAP Value (Log-Odds Impact on f(x))",
            zeroline=True,
            zerolinecolor="#64748b",
            zerolinewidth=2,
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, family="monospace"),
            gridcolor="#1e293b",
        ),
        height=max(360, len(selected) * 26 + 100),
    )
    return fig


def create_metrics_bar_chart(metrics: Dict[str, float]) -> go.Figure:
    """
    Build a summary comparison bar chart of scalar model performance metrics.
    """
    metric_keys = [
        ("Accuracy", "accuracy", "#38bdf8"),
        ("Precision", "precision", "#10b981"),
        ("Recall / TPR", "recall", "#818cf8"),
        ("Specificity", "specificity", "#34d399"),
        ("F1-Score", "f1", "#f59e0b"),
        ("ROC-AUC", "roc_auc", "#a78bfa"),
        ("PR-AUC", "pr_auc", "#ec4899"),
    ]

    labels = []
    values = []
    colors = []

    for label, key, color in metric_keys:
        if key in metrics:
            labels.append(label)
            values.append(float(metrics[key]))
            colors.append(color)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors, line=dict(color="#0d1322", width=1)),
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="<b>Model Performance Metric Summary (Test Split)</b>", font=dict(size=14, color="#f8fafc")),
        yaxis=dict(title="Score", range=[0.0, 1.05], gridcolor="#1e293b"),
        xaxis=dict(title="", tickfont=dict(size=11)),
        height=320,
    )
    return fig


def create_confusion_matrix_heatmap(cm: Dict[str, int]) -> go.Figure:
    """
    Build a styled 2x2 confusion matrix heatmap.
    """
    tn = cm.get("tn", 0)
    fp = cm.get("fp", 0)
    fn = cm.get("fn", 0)
    tp = cm.get("tp", 0)

    z = [[tn, fp], [fn, tp]]
    total = max(1, tn + fp + fn + tp)

    annotations = [
        [f"<b>True Negative (TN)</b><br>{tn}<br>({tn/total:.1%})", f"<b>False Positive (FP)</b><br>{fp}<br>({fp/total:.1%})"],
        [f"<b>False Negative (FN)</b><br>{fn}<br>({fn/total:.1%})", f"<b>True Positive (TP)</b><br>{tp}<br>({tp/total:.1%})"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Benign", "Predicted Malicious"],
            y=["Actual Benign", "Actual Malicious"],
            text=annotations,
            texttemplate="%{text}",
            colorscale=[[0, "#131b2e"], [0.5, "#1e3a5f"], [1, "#0284c7"]],
            showscale=False,
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="<b>Confusion Matrix Heatmap (N = 240)</b>", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(side="bottom", tickfont=dict(size=12, color="#cbd5e1")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#cbd5e1")),
        height=320,
    )
    return fig


def create_threshold_tradeoff_chart(
    threshold_points: List[Dict[str, Any]],
    current_threshold: float = 0.5,
) -> go.Figure:
    """
    Build a multi-metric threshold evaluation trade-off curve.
    """
    thresholds = [p["threshold"] for p in threshold_points]
    tprs = [p["recall"] for p in threshold_points]
    fprs = [p["false_positive_rate"] for p in threshold_points]
    fnrs = [p["false_negative_rate"] for p in threshold_points]
    f1s = [p["f1_score"] for p in threshold_points]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=thresholds, y=tprs, mode="lines+markers", name="TPR / Recall", line=dict(color="#10b981", width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=fnrs, mode="lines+markers", name="FNR (Missed Malware)", line=dict(color="#ef4444", width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=thresholds, y=fprs, mode="lines+markers", name="FPR (Alert Fatigue)", line=dict(color="#f59e0b", width=2.5, dash="dot")))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines+markers", name="F1-Score", line=dict(color="#38bdf8", width=2)))

    # Add active threshold line
    fig.add_vline(
        x=current_threshold,
        line_width=2,
        line_dash="dash",
        line_color="#a855f7",
        annotation_text=f"Selected Cutoff: {current_threshold:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#d8b4fe"),
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text="<b>Operating Metric Trade-offs Across Decision Thresholds</b>",
            font=dict(size=14, color="#f8fafc"),
        ),
        xaxis=dict(title="Decision Threshold (t)", range=[0.0, 1.0], gridcolor="#1e293b"),
        yaxis=dict(title="Metric Score", range=[0.0, 1.05], gridcolor="#1e293b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


def create_group_importance_chart(groups_data: Dict[str, Any]) -> go.Figure:
    """
    Build a bar chart showing mean absolute SHAP attribution across the 8 EMBER groups.
    """
    group_names = []
    group_shaps = []

    for name, data in groups_data.items():
        group_names.append(name.capitalize())
        group_shaps.append(data.get("mean_abs_shap", 0.0))

    # Sort descending
    sorted_pairs = sorted(zip(group_names, group_shaps), key=lambda p: p[1], reverse=True)
    names, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker=dict(
                color=["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#ec4899", "#a78bfa", "#f43f5e", "#10b981"],
                line=dict(color="#0d1322", width=1),
            ),
            text=[f"{v:.4f}" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="<b>Mean Absolute SHAP Attribution by EMBER Feature Group</b>", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="EMBER Schema Group", tickfont=dict(size=11)),
        yaxis=dict(title="Mean |SHAP Value|", gridcolor="#1e293b"),
        height=340,
    )
    return fig


def create_top_features_chart(top_features: List[Dict[str, Any]]) -> go.Figure:
    """
    Build a horizontal bar chart of top 20 global features ranked by mean absolute SHAP.
    """
    top20 = list(top_features[:20])
    top20.reverse()

    names = [f["feature_name"] for f in top20]
    shaps = [f["mean_abs_shap"] for f in top20]
    groups = [f.get("feature_group", "unknown") for f in top20]

    hover_texts = [
        f"<b>{f['feature_name']}</b><br>"
        f"Group: {f.get('feature_group', 'unknown')}<br>"
        f"Mean |SHAP|: {f['mean_abs_shap']:.5f}<br>"
        f"<i>{f.get('description', '')[:70]}...</i>"
        for f in top20
    ]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=shaps,
            orientation="h",
            marker=dict(color="#38bdf8", line=dict(color="#0d1322", width=1)),
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="<b>Top 20 Global Features by Mean Absolute SHAP (E[|phi|])</b>", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="Mean Absolute SHAP Value", gridcolor="#1e293b"),
        yaxis=dict(title="", tickfont=dict(size=10, family="monospace")),
        height=520,
    )
    return fig
