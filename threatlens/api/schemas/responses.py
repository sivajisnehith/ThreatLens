"""
ThreatLens API Response Schemas (Pydantic v2)

Defines strongly-typed response models for health, model metadata, inference,
explanation, and evaluation metric endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from threatlens.explainability.explainer import SYNTHETIC_MODEL_DISCLAIMER
from threatlens.model.calibration import CALIBRATION_SYNTHETIC_DISCLAIMER


class HealthResponse(BaseModel):
    """Health check status and service readiness."""
    status: str = Field(default="healthy", description="Application service health status")
    model_loaded: bool = Field(..., description="Whether LightGBM classifier is loaded")
    model_version: str = Field(..., description="Active model release version")
    model_status: str = Field(default="SYNTHETIC — RESEARCH ONLY", description="Operational model provenance indicator")
    schema_version: str = Field(..., description="Active EMBER/PE feature schema version")
    calibration_status: str = Field(..., description="'calibrated' or 'uncalibrated'")
    calibration_method: Optional[str] = Field(None, description="Calibration algorithm ('sigmoid' or 'isotonic')")
    disclaimer: str = Field(default=SYNTHETIC_MODEL_DISCLAIMER, description="Dataset provenance limitation notice")


class ModelInfoResponse(BaseModel):
    """Detailed model architecture, metadata, and limitations."""
    algorithm: str = Field(..., description="Underlying ML algorithm family")
    feature_dimension: int = Field(default=2351, description="Expected input feature dimension")
    feature_schema: str = Field(default="EMBER_2018", description="Official feature representation schema")
    dataset_provenance: str = Field(..., description="Dataset origin and synthetic data disclosure")
    model_status: str = Field(default="SYNTHETIC — RESEARCH ONLY", description="Operational model provenance indicator")
    model_version: str = Field(..., description="Model artifact version")
    training_metadata: Dict[str, Any] = Field(..., description="Hyperparameters and training metrics")
    calibration_info: Dict[str, Any] = Field(..., description="Fitted calibrator status and metrics")
    limitations: List[str] = Field(..., description="Explicit architectural and security limitations")


class PredictResponse(BaseModel):
    """Inference response for a single PE feature vector."""
    prediction: int = Field(..., description="Discrete classification: 0 (Benign) or 1 (Malicious)")
    prediction_label: str = Field(..., description="Human-readable label ('BENIGN' or 'MALICIOUS')")
    raw_probability: float = Field(..., description="Raw model probability of being malicious")
    calibrated_probability: Optional[float] = Field(None, description="Calibrated probability if calibrator is active")
    threshold: float = Field(..., description="Decision threshold applied")
    confidence: float = Field(..., description="Prediction confidence relative to predicted class")
    model_version: str = Field(..., description="Model version used for inference")
    warning: str = Field(default=SYNTHETIC_MODEL_DISCLAIMER, description="Experimental synthetic model notice")


class BatchPredictResponse(BaseModel):
    """Inference response for a batch of PE feature vectors."""
    count: int = Field(..., description="Total samples in batch")
    predictions: List[int] = Field(..., description="Discrete predictions array")
    prediction_labels: List[str] = Field(..., description="Prediction labels array")
    raw_probabilities: List[float] = Field(..., description="Raw probabilities array")
    calibrated_probabilities: Optional[List[float]] = Field(None, description="Calibrated probabilities array")
    threshold: float = Field(..., description="Decision threshold applied")
    model_version: str = Field(..., description="Model version used")
    warning: str = Field(default=SYNTHETIC_MODEL_DISCLAIMER, description="Experimental notice")


class ExplainResponse(BaseModel):
    """Local SHAP attribution explanation for a single prediction."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    prediction_label: str = Field(..., description="Predicted label name")
    raw_score: float = Field(..., description="Model log-odds margin output f(x)")
    raw_probability: float = Field(..., description="Raw prediction probability")
    calibrated_probability: Optional[float] = Field(None, description="Calibrated probability if available")
    baseline: float = Field(..., description="Baseline expected value E[f(x)] across training set")
    threshold: float = Field(..., description="Decision threshold")
    additivity_error: float = Field(..., description="Exact additivity error |E[f(x)] + sum(SHAP) - raw_score|")
    top_positive_contributors: List[Dict[str, Any]] = Field(..., description="Top features increasing malware risk (+ SHAP)")
    top_negative_contributors: List[Dict[str, Any]] = Field(..., description="Top features decreasing malware risk (- SHAP)")
    warning: str = Field(default=SYNTHETIC_MODEL_DISCLAIMER, description="Experimental attribution notice")


class ModelMetricsResponse(BaseModel):
    """Untouched test set evaluation metrics loaded from Stage 3 artifacts."""
    threatlens_stage: int = Field(default=3, description="Project pipeline stage")
    metrics: Dict[str, Any] = Field(..., description="Evaluation metrics (accuracy, precision, recall, FPR, FNR, etc.)")
    confusion_matrix: Dict[str, int] = Field(..., description="Confusion matrix (tn, fp, fn, tp)")
    dataset_type: str = Field(default="synthetic", description="Provenance of evaluation dataset")
    disclaimer: str = Field(default=SYNTHETIC_MODEL_DISCLAIMER, description="Experimental disclaimer")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    error: str = Field(..., description="Error category / name")
    detail: str = Field(..., description="Detailed error description")
    status_code: int = Field(..., description="HTTP status code")
