"""
ThreatLens Local Explanation Endpoint
"""

from fastapi import APIRouter, Depends, HTTPException, status

from threatlens.api.dependencies import ShapExplainer, get_explainer
from threatlens.api.schemas.requests import ExplainRequest
from threatlens.api.schemas.responses import ExplainResponse
from threatlens.explainability.explainer import SYNTHETIC_MODEL_DISCLAIMER
from threatlens.utils.logger import get_logger

logger = get_logger("api_explanation")
router = APIRouter(tags=["Explainability"])


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Local SHAP Explanation",
    description="Compute mathematical Shapley feature attributions (TreeExplainer) for an individual prediction.",
)
async def explain_prediction(
    payload: ExplainRequest,
    explainer: ShapExplainer = Depends(get_explainer),
) -> ExplainResponse:
    """Compute local SHAP explanation for a 2,351-dimensional EMBER feature vector."""
    try:
        explanation = explainer.explain_single(
            feature_vector=payload.features,
            top_k=payload.top_k,
            threshold=payload.threshold,
        )
    except Exception as e:
        logger.error(f"Explanation error during /explain: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation failed due to an internal error.",
        ) from e

    return ExplainResponse(
        prediction=explanation.prediction,
        prediction_label=explanation.prediction_label,
        raw_score=explanation.raw_score,
        raw_probability=explanation.probability,
        calibrated_probability=explanation.calibrated_probability,
        baseline=explanation.expected_value,
        threshold=explanation.threshold,
        additivity_error=explanation.additivity_error,
        top_positive_contributors=[c.to_dict() for c in explanation.top_positive_contributors],
        top_negative_contributors=[c.to_dict() for c in explanation.top_negative_contributors],
        warning=SYNTHETIC_MODEL_DISCLAIMER,
    )
