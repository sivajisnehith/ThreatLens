"""
ThreatLens Model Information and Metrics Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status

from threatlens.api.dependencies import ModelContainer, get_model_container
from threatlens.api.schemas.responses import ModelInfoResponse, ModelMetricsResponse
from threatlens.explainability.explainer import SYNTHETIC_MODEL_DISCLAIMER

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Architecture & Metadata",
    description="Returns model architecture specifications, training hyperparameters, calibration info, and limitations.",
)
async def get_model_info(
    container: ModelContainer = Depends(get_model_container),
) -> ModelInfoResponse:
    """Return comprehensive metadata for the active LightGBM malware classifier."""
    if not container.is_ready or container.classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable.",
        )

    metadata = container.classifier.get_metadata()
    model_version = "1.0.0"
    if metadata:
        model_version = f"{metadata.get('threatlens_stage', 3)}.0.0"

    cal_info = {
        "is_calibrated": False,
        "method": None,
        "status": "uncalibrated",
    }
    if container.calibrator and container.calibrator.is_fitted:
        cal_info = {
            "is_calibrated": True,
            "method": container.calibrator.method,
            "status": "calibrated",
            "metadata": container.calibrator.metadata,
        }

    prov = str(metadata.get("dataset_provenance", "synthetic")).lower()
    if "genuine" in prov:
        model_status = "GENUINE DATASET — EVALUATION MODEL"
        limitations = [
            "STATIC ANALYSIS ONLY: Features are extracted from PE headers and static byte distributions without dynamic execution. Packing, polymorphism, or crypters may obscure malicious intent.",
            "FEATURE HASHING: EMBER uses MurmurHash for imports (1,280 bins), exports (128 bins), and sections. Hashed bins cannot be uniquely mapped back to raw API names due to hash collisions.",
            "NON-CAUSALITY: Feature importance indicates tree split usage, not causal malware mechanisms.",
            "TEMPORAL DRIFT: EMBER 2018 dataset represents PE binaries from 2018; newer malware families and compiler tooling may exhibit distribution shift.",
        ]
    else:
        model_status = "SYNTHETIC — RESEARCH ONLY"
        limitations = [
            "SYNTHETIC DATA: The current model was trained on synthetic data for pipeline validation. Probabilities and SHAP attributions reflect synthetic distributions and do not represent verified cybersecurity intelligence.",
            "FEATURE HASHING: EMBER uses MurmurHash for imports (1,248 bins), exports (68 bins), and sections. Hashed bins cannot be uniquely mapped back to raw API names due to hash collisions.",
            "NON-CAUSALITY: Feature importance indicates tree split usage, not causal malware mechanisms.",
            "CORRELATION: Strongly correlated features may share attribution across trees.",
        ]

    return ModelInfoResponse(
        algorithm="LightGBM Gradient Boosted Decision Trees (GBDT)",
        feature_dimension=container.classifier.expected_dim,
        feature_schema="EMBER_2018",
        dataset_provenance=metadata.get("provenance_notes", "Stage 1 EMBER 2018 synthetic schema dataset"),
        model_status=model_status,
        model_version=model_version,
        training_metadata=metadata,
        calibration_info=cal_info,
        limitations=limitations,
    )


@router.get(
    "/metrics/model",
    response_model=ModelMetricsResponse,
    summary="Model Evaluation Metrics",
    description="Returns untouched test set evaluation metrics loaded from serialized evaluation reports.",
)
async def get_model_metrics(
    container: ModelContainer = Depends(get_model_container),
) -> ModelMetricsResponse:
    """Return test evaluation metrics from saved reports. Does NOT evaluate on calibration split."""
    report = container.evaluation_report
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation metrics artifact (classification_report.json) not found on disk.",
        )

    is_genuine = "genuine" in str(report.get("dataset_provenance", "")).lower()
    dataset_type = "genuine_ember_2018_test_split" if is_genuine else "synthetic_ember_test_split"
    disclaimer = (
        "Official EMBER 2018 test split benchmark. Evaluated on genuine static malware features for research and evaluation."
        if is_genuine
        else SYNTHETIC_MODEL_DISCLAIMER
    )

    return ModelMetricsResponse(
        threatlens_stage=report.get("threatlens_stage", 3),
        metrics=report.get("metrics", {}),
        confusion_matrix=report.get("confusion_matrix", {}),
        dataset_type=dataset_type,
        disclaimer=disclaimer,
    )
