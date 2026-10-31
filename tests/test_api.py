"""
ThreatLens Stage 5: FastAPI Backend Test Suite

Tests all REST API endpoints:
- GET /health
- GET /model/info
- GET /metrics/model
- POST /predict
- POST /predict/batch
- POST /explain

Validates strict schema constraints, rejection of NaN/Inf values, batch limits,
and error handling.
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from threatlens.api.main import app
from threatlens.config import EmberFeatureGroups

EXPECTED_DIM = EmberFeatureGroups.TOTAL_FEATURES  # 2351


@pytest.fixture(scope="module")
def client():
    """Create reusable FastAPI TestClient with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_feature_vector():
    """Generate a clean, deterministic 2,351-dimensional feature vector."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, size=EXPECTED_DIM).astype(float).tolist()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check_returns_200(self, client):
        """Verify GET /health returns 200 and valid schema."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["model_loaded"] is True
        assert data["schema_version"] == "1.0.0"
        assert data["calibration_status"] in ("calibrated", "uncalibrated")
        assert "synthetic" in data["disclaimer"].lower()


class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info_returns_200(self, client):
        """Verify GET /model/info returns metadata and limitations."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "LightGBM" in data["algorithm"]
        assert data["feature_dimension"] == EXPECTED_DIM
        assert data["feature_schema"] == "EMBER_2018"
        assert isinstance(data["training_metadata"], dict)
        assert isinstance(data["limitations"], list)
        assert len(data["limitations"]) >= 3


class TestModelMetricsEndpoint:
    """Tests for GET /metrics/model."""

    def test_model_metrics_returns_200(self, client):
        """Verify GET /metrics/model loads saved Stage 3 evaluation metrics."""
        response = client.get("/metrics/model")
        assert response.status_code == 200

        data = response.json()
        assert data["threatlens_stage"] in (3, 7)
        assert "accuracy" in data["metrics"]
        assert "precision" in data["metrics"]
        assert "recall" in data["metrics"]
        assert "confusion_matrix" in data


class TestPredictionEndpoints:
    """Tests for POST /predict and POST /predict/batch."""

    def test_predict_single_success(self, client, valid_feature_vector):
        """Verify POST /predict returns valid inference payload."""
        payload = {
            "features": valid_feature_vector,
            "threshold": 0.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["prediction"] in (0, 1)
        assert data["prediction_label"] in ("BENIGN", "MALICIOUS")
        assert 0.0 <= data["raw_probability"] <= 1.0
        assert data["threshold"] == 0.5
        assert 0.0 <= data["confidence"] <= 1.0
        assert "warning" in data

    def test_predict_returns_calibrated_probability_if_available(self, client, valid_feature_vector):
        """Verify calibrated probability is included and within [0, 1]."""
        response = client.post("/predict", json={"features": valid_feature_vector})
        assert response.status_code == 200

        data = response.json()
        if data.get("calibrated_probability") is not None:
            assert 0.0 <= data["calibrated_probability"] <= 1.0

    def test_predict_threshold_adjustment(self, client, valid_feature_vector):
        """Verify custom threshold influences discrete classification."""
        # Extremely high threshold -> Benign
        resp_high = client.post("/predict", json={"features": valid_feature_vector, "threshold": 0.999})
        assert resp_high.status_code == 200
        assert resp_high.json()["prediction"] == 0
        assert resp_high.json()["prediction_label"] == "BENIGN"

        # Extremely low threshold -> Malicious
        resp_low = client.post("/predict", json={"features": valid_feature_vector, "threshold": 0.001})
        assert resp_low.status_code == 200
        assert resp_low.json()["prediction"] == 1
        assert resp_low.json()["prediction_label"] == "MALICIOUS"

    def test_batch_predict_success(self, client, valid_feature_vector):
        """Verify POST /predict/batch processes multiple samples."""
        batch = [valid_feature_vector, valid_feature_vector, valid_feature_vector]
        payload = {"samples": batch, "threshold": 0.5}

        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3
        assert len(data["prediction_labels"]) == 3
        assert len(data["raw_probabilities"]) == 3


class TestExplainEndpoint:
    """Tests for POST /explain."""

    def test_explain_single_success(self, client, valid_feature_vector):
        """Verify POST /explain returns local SHAP attributions."""
        payload = {
            "features": valid_feature_vector,
            "top_k": 5,
            "threshold": 0.5,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["prediction"] in (0, 1)
        assert "raw_score" in data
        assert "baseline" in data
        assert "additivity_error" in data
        assert len(data["top_positive_contributors"]) <= 5
        assert len(data["top_negative_contributors"]) <= 5


class TestSecurityAndInputValidation:
    """Security tests: strict validation, rejection of NaN/Inf, and batch limits."""

    def test_rejects_wrong_feature_dimension(self, client):
        """Verify vectors with len != 2351 are rejected with HTTP 422."""
        # 85-dimensional custom PE features accidentally sent to EMBER endpoint
        payload = {"features": [0.0] * 85}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "dimension mismatch" in response.text.lower()

    def test_rejects_nan_values(self, client, valid_feature_vector):
        """Verify feature vector containing NaN is strictly rejected with HTTP 422."""
        import json
        bad_vector = list(valid_feature_vector)
        bad_vector[10] = float("nan")

        raw_body = json.dumps({"features": bad_vector}, allow_nan=True)
        response = client.post("/predict", content=raw_body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert "nan" in response.text.lower()

    def test_rejects_infinite_values(self, client, valid_feature_vector):
        """Verify feature vector containing Inf is strictly rejected with HTTP 422."""
        import json
        bad_vector = list(valid_feature_vector)
        bad_vector[20] = float("inf")

        raw_body = json.dumps({"features": bad_vector}, allow_nan=True)
        response = client.post("/predict", content=raw_body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert "infinite" in response.text.lower()

    def test_rejects_oversized_batch(self, client, valid_feature_vector):
        """Verify batches exceeding limit of 100 samples are rejected with HTTP 422."""
        oversized = [valid_feature_vector] * 105
        response = client.post("/predict/batch", json={"samples": oversized})
        assert response.status_code == 422
        assert "batch size" in response.text.lower() or "too_long" in response.text.lower()

    def test_rejects_invalid_threshold_range(self, client, valid_feature_vector):
        """Verify thresholds outside [0.0, 1.0] are rejected with HTTP 422."""
        resp_too_high = client.post("/predict", json={"features": valid_feature_vector, "threshold": 1.5})
        assert resp_too_high.status_code == 422

        resp_too_low = client.post("/predict", json={"features": valid_feature_vector, "threshold": -0.2})
        assert resp_too_low.status_code == 422

    def test_rejects_empty_payload(self, client):
        """Verify empty JSON request is rejected with HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
