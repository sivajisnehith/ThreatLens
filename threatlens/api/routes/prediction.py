"""
ThreatLens Prediction Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np

from threatlens.api.dependencies import MalwareClassifier, get_classifier
from threatlens.api.schemas.requests import BatchPredictRequest, PredictRequest
from threatlens.api.schemas.responses import BatchPredictResponse, PredictResponse
from threatlens.explainability.explainer import SYNTHETIC_MODEL_DISCLAIMER
from threatlens.utils.logger import get_logger

logger = get_logger("api_prediction")
router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single Sample Prediction",
    description="Predict binary malware classification for a validated 2,351-dimensional EMBER feature vector.",
)
async def predict_single(
    payload: PredictRequest,
    classifier: MalwareClassifier = Depends(get_classifier),
) -> PredictResponse:
    """Execute validated binary inference on a single feature vector."""
    try:
        # Run inference using MalwareClassifier
        result = classifier.predict_single(payload.features, threshold=payload.threshold)
    except Exception as e:
        logger.error(f"Inference error during /predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an internal error.",
        ) from e

    model_ver = "1.0.0"
    is_genuine = False
    if classifier.metadata:
        model_ver = f"{classifier.metadata.get('threatlens_stage', 3)}.0.0"
        is_genuine = "genuine" in str(classifier.metadata.get("dataset_provenance", "")).lower()

    warning_notice = (
        "Official EMBER 2018 benchmark evaluation model. Static analysis features for research and evaluation."
        if is_genuine
        else SYNTHETIC_MODEL_DISCLAIMER
    )

    return PredictResponse(
        prediction=result.label,
        prediction_label=result.label_name,
        raw_probability=result.probability,
        calibrated_probability=result.calibrated_probability,
        threshold=result.threshold,
        confidence=result.confidence,
        model_version=model_ver,
        warning=warning_notice,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch Sample Prediction",
    description="Perform high-performance batch inference on multiple 2,351-dimensional EMBER feature vectors (up to 100).",
)
async def predict_batch(
    payload: BatchPredictRequest,
    classifier: MalwareClassifier = Depends(get_classifier),
) -> BatchPredictResponse:
    """Execute batch binary inference on multiple feature vectors."""
    try:
        batch_result = classifier.predict_batch(payload.samples, threshold=payload.threshold)
    except Exception as e:
        logger.error(f"Inference error during /predict/batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference failed due to an internal error.",
        ) from e

    labels_list = batch_result.labels.tolist()
    label_names = ["MALICIOUS" if l == 1 else "BENIGN" for l in labels_list]
    raw_probs_list = [float(p) for p in batch_result.probabilities]

    cal_probs_list = None
    if batch_result.calibrated_probabilities is not None:
        cal_probs_list = [float(p) for p in batch_result.calibrated_probabilities]

    model_ver = "1.0.0"
    is_genuine = False
    if classifier.metadata:
        model_ver = f"{classifier.metadata.get('threatlens_stage', 3)}.0.0"
        is_genuine = "genuine" in str(classifier.metadata.get("dataset_provenance", "")).lower()

    warning_notice = (
        "Official EMBER 2018 benchmark evaluation model. Static analysis features for research and evaluation."
        if is_genuine
        else SYNTHETIC_MODEL_DISCLAIMER
    )

    return BatchPredictResponse(
        count=batch_result.count,
        predictions=labels_list,
        prediction_labels=label_names,
        raw_probabilities=raw_probs_list,
        calibrated_probabilities=cal_probs_list,
        threshold=payload.threshold,
        model_version=model_ver,
        warning=warning_notice,
    )
