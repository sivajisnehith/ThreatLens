"""
ThreatLens API Request Schemas (Pydantic v2)

Strict request validation for inference, explanation, and batch operations.
Enforces 2,351 dimensions, rejects NaN/Inf values, and enforces batch limits.
"""

import math
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from threatlens.config import API_CONFIG, EmberFeatureGroups

EXPECTED_DIM = EmberFeatureGroups.TOTAL_FEATURES  # 2351
MAX_BATCH_SIZE = API_CONFIG.MAX_BATCH_SIZE        # 100


def _validate_vector(features: List[float]) -> List[float]:
    """Validate 2,351-dimensional numerical vector and reject NaN / Inf."""
    if len(features) != EXPECTED_DIM:
        raise ValueError(
            f"Feature dimension mismatch: expected exactly {EXPECTED_DIM} EMBER features, got {len(features)}"
        )

    for i, val in enumerate(features):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Feature at index {i} must be a number, got {type(val).__name__}")
        if math.isnan(val):
            raise ValueError(f"Feature vector contains NaN at index {i}. NaN inputs are strictly rejected.")
        if math.isinf(val):
            raise ValueError(f"Feature vector contains infinite value at index {i}. Infinite inputs are strictly rejected.")

    return features


class PredictRequest(BaseModel):
    """Request payload for single sample inference."""
    features: List[float] = Field(
        ...,
        description="2,351-dimensional EMBER 2018 feature vector",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for malicious classification (default 0.50)",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        return _validate_vector(v)


class BatchPredictRequest(BaseModel):
    """Request payload for batch inference."""
    samples: List[List[float]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of feature vectors (batch size limit: {MAX_BATCH_SIZE})",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for malicious classification (default 0.50)",
    )

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum allowed limit of {MAX_BATCH_SIZE}")

        for idx, sample in enumerate(v):
            try:
                _validate_vector(sample)
            except ValueError as e:
                raise ValueError(f"Validation failure in batch sample index {idx}: {e}") from e

        return v


class ExplainRequest(BaseModel):
    """Request payload for local SHAP explanation."""
    features: List[float] = Field(
        ...,
        description="2,351-dimensional EMBER 2018 feature vector",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of top positive and negative influential features to return (1-100)",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for malicious classification (default 0.50)",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        return _validate_vector(v)
