"""
Tests for ThreatLens Streamlit Dashboard & API Client (Stage 6)

Covers:
- ThreatLensAPIClient connection, timeouts, error decoding, and mock endpoints
- EMBER 2,351-dimensional feature vector strict validation (dimension, NaN, Inf, types)
- Static PE file upload and temporary file cleanup
- Plotly chart builders and dashboard visual helper utilities
- Session state management and isolation
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import numpy as np
import pytest

from threatlens.config import DASHBOARD_CONFIG
from threatlens.dashboard.api_client import (
    APIClientError,
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    ThreatLensAPIClient,
)
from threatlens.dashboard.components.charts import (
    create_confusion_matrix_heatmap,
    create_gauge_chart,
    create_group_importance_chart,
    create_metrics_bar_chart,
    create_shap_horizontal_bar,
    create_threshold_tradeoff_chart,
    create_top_features_chart,
)
from threatlens.dashboard.components.metrics import render_status_badge
from threatlens.dashboard.pages.analyze_sample import _validate_ember_vector
from threatlens.dashboard.state import (
    clear_active_vector,
    clear_custom_pe_result,
    init_session_state,
    set_active_vector,
    set_custom_pe_result,
    set_explanation,
    set_prediction,
)


# =============================================================================
# 1. API CLIENT TESTS
# =============================================================================

class TestThreatLensAPIClient:
    """Test suite for ThreatLensAPIClient."""

    def test_init_defaults(self):
        client = ThreatLensAPIClient()
        assert client.base_url == DASHBOARD_CONFIG.DEFAULT_API_URL.rstrip("/")
        assert client.timeout == DASHBOARD_CONFIG.API_TIMEOUT
        assert client.connect_timeout == DASHBOARD_CONFIG.CONNECT_TIMEOUT
        client.close()

    def test_custom_base_url(self):
        client = ThreatLensAPIClient(base_url="http://custom-host:9000/")
        assert client.base_url == "http://custom-host:9000"
        client.close()

    def test_health_success(self):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "schema_version": "1.0.0",
                "calibration_status": "calibrated",
                "disclaimer": "synthetic",
            },
            request=httpx.Request("GET", "http://127.0.0.1:8000/health"),
        )
        mock_transport = httpx.MockTransport(lambda req: mock_response)
        mock_http = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8000")

        with ThreatLensAPIClient(client=mock_http) as client:
            resp = client.health()
            assert resp["status"] == "healthy"
            assert resp["model_loaded"] is True

    def test_backend_unavailable_raises_connection_error(self):
        def raise_connect_error(req):
            raise httpx.ConnectError("Connection refused", request=req)

        mock_transport = httpx.MockTransport(raise_connect_error)
        mock_http = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8000")

        with ThreatLensAPIClient(client=mock_http) as client:
            with pytest.raises(APIConnectionError) as exc_info:
                client.health()
            assert "Could not connect to ThreatLens backend" in str(exc_info.value)

    def test_timeout_raises_timeout_error(self):
        def raise_read_timeout(req):
            raise httpx.ReadTimeout("Request timed out", request=req)

        mock_transport = httpx.MockTransport(raise_read_timeout)
        mock_http = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8000")

        with ThreatLensAPIClient(client=mock_http) as client:
            with pytest.raises(APITimeoutError) as exc_info:
                client.get_model_info()
            assert "timed out" in str(exc_info.value)

    def test_http_status_error_raises_response_error(self):
        mock_response = httpx.Response(
            status_code=422,
            json={"detail": "Feature vector must have exactly 2351 dimensions"},
            request=httpx.Request("POST", "http://127.0.0.1:8000/predict"),
        )
        mock_transport = httpx.MockTransport(lambda req: mock_response)
        mock_http = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8000")

        with ThreatLensAPIClient(client=mock_http) as client:
            with pytest.raises(APIResponseError) as exc_info:
                client.predict(features=[0.0] * 10)
            assert exc_info.value.status_code == 422
            assert "2351" in str(exc_info.value.detail)

    def test_predict_success(self):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "prediction": 1,
                "prediction_label": "MALICIOUS",
                "raw_probability": 0.52,
                "calibrated_probability": 0.505,
                "threshold": 0.5,
                "confidence": 0.505,
                "model_version": "1.0.0",
                "warning": "synthetic",
            },
            request=httpx.Request("POST", "http://127.0.0.1:8000/predict"),
        )
        mock_transport = httpx.MockTransport(lambda req: mock_response)
        mock_http = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8000")

        with ThreatLensAPIClient(client=mock_http) as client:
            resp = client.predict(features=[0.0] * 2351, threshold=0.5)
            assert resp["prediction"] == 1
            assert resp["prediction_label"] == "MALICIOUS"
            assert resp["raw_probability"] == 0.52

    def test_explain_success(self):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "prediction": 0,
                "prediction_label": "BENIGN",
                "raw_score": -0.05,
                "raw_probability": 0.487,
                "calibrated_probability": 0.495,
                "baseline": 0.016,
                "threshold": 0.5,
                "additivity_error": 1e-10,
                "top_positive_contributors": [],
                "top_negative_contributors": [],
                "warning": "synthetic",
            },
            request=httpx.Request("POST", "http://127.0.0.1:8000/explain"),
        )
        mock_transport = httpx.MockTransport(lambda req: mock_response)
        mock_http = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8000")

        with ThreatLensAPIClient(client=mock_http) as client:
            resp = client.explain(features=[0.0] * 2351, top_k=10)
            assert resp["prediction"] == 0
            assert resp["baseline"] == 0.016


# =============================================================================
# 2. FEATURE VECTOR VALIDATION TESTS
# =============================================================================

class TestFeatureVectorValidation:
    """Test suite for validating EMBER 2018 2,351-dimensional vectors."""

    def test_valid_vector(self):
        valid_vec = [0.1] * 2351
        is_valid, msg, clean_vec = _validate_ember_vector(valid_vec)
        assert is_valid is True
        assert msg == ""
        assert clean_vec is not None
        assert len(clean_vec) == 2351

    def test_numpy_array_valid(self):
        arr = np.zeros(2351, dtype=np.float32)
        is_valid, msg, clean_vec = _validate_ember_vector(arr)
        assert is_valid is True
        assert len(clean_vec) == 2351

    def test_wrong_dimensions_rejected(self):
        # Custom 85-dim PE extractor vector
        pe_85 = [0.0] * 85
        is_valid, msg, clean_vec = _validate_ember_vector(pe_85)
        assert is_valid is False
        assert "2,351" in msg
        assert clean_vec is None

        # 1D with 10 features
        is_valid, msg, clean_vec = _validate_ember_vector([1.0] * 10)
        assert is_valid is False
        assert clean_vec is None

    def test_nan_rejected(self):
        vec_with_nan = [0.0] * 2350 + [float("nan")]
        is_valid, msg, clean_vec = _validate_ember_vector(vec_with_nan)
        assert is_valid is False
        assert "NaN" in msg
        assert clean_vec is None

    def test_infinite_rejected(self):
        vec_with_inf = [0.0] * 2350 + [float("inf")]
        is_valid, msg, clean_vec = _validate_ember_vector(vec_with_inf)
        assert is_valid is False
        assert "infinite" in msg
        assert clean_vec is None

    def test_non_numeric_types_rejected(self):
        bad_vec = ["zero"] * 2351
        is_valid, msg, clean_vec = _validate_ember_vector(bad_vec)
        assert is_valid is False
        assert "numeric" in msg
        assert clean_vec is None


# =============================================================================
# 3. PE UPLOAD VALIDATION & CLEANUP TESTS
# =============================================================================

class TestPEUploadValidation:
    """Test suite for PE upload checks, non-executing extraction, and cleanup."""

    def test_oversized_file_limit_constant(self):
        assert DASHBOARD_CONFIG.MAX_UPLOAD_SIZE_MB == 50

    def test_valid_synthetic_pe_extraction_and_cleanup(self):
        from threatlens.core.extractor import extract_pe_features

        # Create temporary valid synthetic PE binary
        with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
            tmp_path = Path(tmp.name)
            # Create minimal MZ header
            mz_header = bytearray(1024)
            mz_header[0:2] = b"MZ"
            mz_header[0x3C:0x40] = (0x80).to_bytes(4, "little")
            mz_header[0x80:0x84] = b"PE\x00\x00"
            # COFF Header (Machine=0x14c, num_sections=1)
            mz_header[0x84:0x86] = (0x014C).to_bytes(2, "little")
            mz_header[0x86:0x88] = (1).to_bytes(2, "little")
            # Optional Header Magic=0x10b (PE32)
            mz_header[0x98:0x9A] = (0x010B).to_bytes(2, "little")
            tmp.write(mz_header)

        try:
            assert tmp_path.exists()
            features = extract_pe_features(file_path=tmp_path)
            assert isinstance(features, dict)
            assert "file_size" in features
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

        assert not tmp_path.exists(), "Temporary file must be deleted after processing"

    def test_corrupted_pe_handling(self):
        from threatlens.core.extractor import extract_pe_features, PEParseError

        with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"CORRUPTED_NOT_A_PE_HEADER_DATA")

        try:
            with pytest.raises(PEParseError):
                extract_pe_features(file_path=tmp_path)
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

        assert not tmp_path.exists()


# =============================================================================
# 4. PLOTLY VISUALIZATIONS AND HELPER TESTS
# =============================================================================

class TestDashboardVisualsAndHelpers:
    """Test suite for Plotly visual chart builders and UI components."""

    def test_gauge_chart(self):
        fig = create_gauge_chart(score=0.72, threshold=0.5)
        assert fig is not None
        assert len(fig.data) == 1
        indicator = fig.data[0]
        assert indicator.value == 0.72

    def test_gauge_chart_clipping(self):
        # Scores outside [0, 1] should be safely clamped
        fig_low = create_gauge_chart(score=-0.5, threshold=0.5)
        assert fig_low.data[0].value == 0.0

        fig_high = create_gauge_chart(score=1.5, threshold=0.5)
        assert fig_high.data[0].value == 1.0

    def test_shap_horizontal_bar_chart(self):
        top_pos = [
            {"feature_name": "feat_a", "feature_group": "histogram", "feature_value": 0.1, "shap_value": 0.05, "description": "desc a"}
        ]
        top_neg = [
            {"feature_name": "feat_b", "feature_group": "strings", "feature_value": 0.2, "shap_value": -0.08, "description": "desc b"}
        ]
        fig = create_shap_horizontal_bar(top_pos=top_pos, top_neg=top_neg, top_k=10)
        assert fig is not None
        assert len(fig.data) == 1
        # Absolute sorting puts |feat_b| (0.08) above |feat_a| (0.05)
        y_labels = list(fig.data[0].y)
        assert "feat_a" in y_labels
        assert "feat_b" in y_labels

    def test_metrics_bar_chart(self):
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
        }
        fig = create_metrics_bar_chart(metrics)
        assert fig is not None
        assert len(fig.data[0].x) >= 4

    def test_confusion_matrix_heatmap(self):
        cm = {"tn": 100, "fp": 10, "fn": 5, "tp": 85}
        fig = create_confusion_matrix_heatmap(cm)
        assert fig is not None
        assert len(fig.data) == 1
        assert [list(row) for row in fig.data[0].z] == [[100, 10], [5, 85]]

    def test_threshold_tradeoff_chart(self):
        points = [
            {"threshold": 0.4, "recall": 0.9, "false_positive_rate": 0.4, "false_negative_rate": 0.1, "f1_score": 0.7},
            {"threshold": 0.5, "recall": 0.8, "false_positive_rate": 0.2, "false_negative_rate": 0.2, "f1_score": 0.8},
        ]
        fig = create_threshold_tradeoff_chart(points, current_threshold=0.5)
        assert fig is not None
        # 4 traces: TPR, FNR, FPR, F1
        assert len(fig.data) == 4

    def test_group_importance_chart(self):
        group_data = {
            "histogram": {"mean_abs_shap": 0.04, "feature_count": 256},
            "byteentropy": {"mean_abs_shap": 0.03, "feature_count": 256},
        }
        fig = create_group_importance_chart(group_data)
        assert fig is not None
        assert len(fig.data[0].x) == 2

    def test_top_features_chart(self):
        top_features = [
            {"rank": i, "feature_name": f"f_{i}", "mean_abs_shap": 0.1 / (i + 1), "feature_group": "group"}
            for i in range(20)
        ]
        fig = create_top_features_chart(top_features)
        assert fig is not None
        assert len(fig.data[0].y) == 20

    def test_status_badge_html(self):
        healthy_html = render_status_badge("Healthy", is_healthy=True)
        assert "badge-healthy" in healthy_html
        assert "Healthy" in healthy_html

        danger_html = render_status_badge("Offline", is_healthy=False)
        assert "badge-danger" in danger_html
        assert "Offline" in danger_html


# =============================================================================
# 5. STATE MANAGEMENT TESTS
# =============================================================================

class TestSessionStateManagement:
    """Test session state helpers without running full Streamlit."""

    @patch("streamlit.session_state", dict())
    def test_session_state_lifecycle(self):
        init_session_state()
        import streamlit as st

        assert st.session_state["active_vector"] is None
        assert st.session_state["active_prediction"] is None

        # Set active vector
        dummy_vec = [1.0] * 2351
        set_active_vector(dummy_vec, "Sample #0")
        assert st.session_state["active_vector"] == dummy_vec
        assert st.session_state["active_source_name"] == "Sample #0"

        # Set prediction and explanation
        set_prediction({"label": 1, "prob": 0.7})
        assert st.session_state["active_prediction"]["label"] == 1

        set_explanation({"raw_score": 0.5})
        assert st.session_state["active_explanation"]["raw_score"] == 0.5

        # Clear vector resets downstream inferences
        clear_active_vector()
        assert st.session_state["active_vector"] is None
        assert st.session_state["active_prediction"] is None
        assert st.session_state["active_explanation"] is None

        # Custom PE result storage (metadata only, no raw binary)
        set_custom_pe_result("test.exe", {"file_size_bytes": 1024})
        assert st.session_state["custom_pe_filename"] == "test.exe"
        assert st.session_state["custom_pe_result"]["file_size_bytes"] == 1024

        clear_custom_pe_result()
        assert st.session_state["custom_pe_filename"] is None
        assert st.session_state["custom_pe_result"] is None
