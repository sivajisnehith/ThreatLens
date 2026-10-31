"""
ThreatLens Health Check Endpoint
"""

from fastapi import APIRouter, Depends

from threatlens.api.dependencies import ModelContainer, get_model_container
from threatlens.api.schemas.responses import HealthResponse
from threatlens.core.schema import SCHEMA_VERSION
from threatlens.explainability.explainer import SYNTHETIC_MODEL_DISCLAIMER

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns API service health, active model status, and calibration readiness.",
)
async def health_check(
    container: ModelContainer = Depends(get_model_container),
) -> HealthResponse:
    """Return application operational health and model loading status."""
    is_loaded = container.is_ready and container.classifier is not None and container.classifier.is_loaded

    cal_status = "uncalibrated"
    cal_method = None
    if container.calibrator is not None and container.calibrator.is_fitted:
        cal_status = "calibrated"
        cal_method = container.calibrator.method

    model_version = "1.0.0"
    model_status = "MODEL UNAVAILABLE"
    if is_loaded:
        prov = str(container.classifier.metadata.get("dataset_provenance", "synthetic")).lower()
        if "genuine" in prov:
            model_status = "GENUINE DATASET — EVALUATION MODEL"
        else:
            model_status = "SYNTHETIC — RESEARCH ONLY"

    if container.classifier and container.classifier.metadata:
        model_version = str(container.classifier.metadata.get("threatlens_stage", "3")) + ".0.0"

    disclaimer = (
        "Official EMBER 2018 benchmark evaluation model. Static analysis features for research and evaluation."
        if "genuine" in prov
        else SYNTHETIC_MODEL_DISCLAIMER
    )

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=model_version,
        model_status=model_status,
        schema_version=SCHEMA_VERSION,
        calibration_status=cal_status,
        calibration_method=cal_method,
        disclaimer=disclaimer,
    )
