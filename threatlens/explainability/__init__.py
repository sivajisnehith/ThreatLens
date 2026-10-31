"""
ThreatLens Explainability Module (Stage 4)

Provides SHAP TreeExplainer integration for LightGBM models trained on the EMBER 2018
schema. Supports single-sample local explanations, batch evaluations, global feature
importance metrics, EMBER group-level aggregations, and publication visualizations.
"""

from threatlens.explainability.explainer import (
    SYNTHETIC_MODEL_DISCLAIMER,
    BatchExplanation,
    ExplainerModelError,
    ExplainerNotInitializedError,
    InvalidFeatureError,
    ShapExplainer,
    SingleExplanation,
)
from threatlens.explainability.feature_interpreter import (
    DIRECTION_DECREASES_MALWARE,
    DIRECTION_INCREASES_MALWARE,
    DIRECTION_NEUTRAL,
    FeatureContribution,
    FeatureInterpreter,
)
from threatlens.explainability.global_explanations import (
    GlobalExplainer,
    GlobalImportanceResult,
    GroupImportanceMetric,
    METRIC_DISTINCTION_EXPLANATION,
)

__all__ = [
    "ShapExplainer",
    "FeatureInterpreter",
    "GlobalExplainer",
    "SingleExplanation",
    "BatchExplanation",
    "FeatureContribution",
    "GlobalImportanceResult",
    "GroupImportanceMetric",
    "ExplainerNotInitializedError",
    "ExplainerModelError",
    "InvalidFeatureError",
    "DIRECTION_INCREASES_MALWARE",
    "DIRECTION_DECREASES_MALWARE",
    "DIRECTION_NEUTRAL",
    "SYNTHETIC_MODEL_DISCLAIMER",
    "METRIC_DISTINCTION_EXPLANATION",
]
