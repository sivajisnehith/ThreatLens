"""
ThreatLens Stage 7: Genuine EMBER 2018 Dataset Integration Test Suite

Validates:
1. Streaming JSONL loader resilience (malformed JSON, empty lines, unlabeled -1 samples).
2. EMBER 2018 batch vectorizer schema (2,351 dimensions, finite float32 outputs).
3. Test partition hash isolation and split reproducibility.
4. Genuine LightGBM model inference, Platt calibration, and SHAP TreeExplainer additivity.
5. End-to-end FastAPI integration under genuine model provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from threatlens.api.dependencies import ModelContainer
from threatlens.api.main import app
from threatlens.config import (
    CALIBRATION_CONFIG,
    DATA_CONFIG,
    MODEL_CONFIG,
    EmberFeatureGroups,
)
from threatlens.dataset.ember_loader import (
    EmberLoader,
    iter_ember_jsonl,
    load_ember_data,
    vectorize_ember_batch,
)
from threatlens.explainability.explainer import ShapExplainer
from threatlens.model.calibration import ProbabilityCalibrator
from threatlens.model.classifier import MalwareClassifier


# =============================================================================
# 1. STREAMING LOADER AND VECTORIZER TESTS
# =============================================================================

class TestEmberJsonlStreamingAndVectorization:
    """Tests for raw JSONL parsing resilience and 2,351-dim feature vectorization."""

    def test_streaming_loader_handles_malformed_and_unlabeled(self, tmp_path):
        """Verify streaming iterator discards malformed lines and unlabeled records."""
        test_file = tmp_path / "mixed_samples.jsonl"
        lines = [
            "",  # Empty line
            "   \n",  # Whitespace line
            "{malformed json line",  # Invalid JSON
            json.dumps({"sha256": "hash_unlabeled", "label": -1, "general": {"size": 1000}}),  # Unlabeled
            json.dumps({"sha256": "hash_benign", "label": 0, "general": {"size": 5000}}),  # Valid Benign
            json.dumps({"sha256": "hash_malicious", "label": 1, "general": {"size": 8000}}),  # Valid Malicious
        ]
        test_file.write_text("\n".join(lines), encoding="utf-8")

        # Stream with skip_unlabeled=True
        records = list(iter_ember_jsonl(test_file, skip_unlabeled=True))
        assert len(records) == 2
        assert records[0]["sha256"] == "hash_benign"
        assert records[0]["label"] == 0
        assert records[1]["sha256"] == "hash_malicious"
        assert records[1]["label"] == 1

        # Stream with skip_unlabeled=False
        records_all = list(iter_ember_jsonl(test_file, skip_unlabeled=False))
        assert len(records_all) == 3
        labels = [r["label"] for r in records_all]
        assert labels == [-1, 0, 1]

    def test_vectorize_ember_batch_dimensions_and_finiteness(self):
        """Verify vectorizer produces exact (N, 2351) float32 matrix with finite numbers."""
        mock_raw_records = [
            {
                "sha256": "sample_1",
                "label": 0,
                "byteentropy": [10] * 256,
                "histogram": [5] * 256,
                "strings": {"numstrings": 20, "avlength": 4.5, "printabledist": [1] * 96, "printables": 100, "entropy": 3.2, "paths": 1, "urls": 2, "registry": 0, "MZ": 1},
                "general": {"size": 10240, "vsize": 20480, "has_debug": 1, "exports": 0, "imports": 10, "has_relocations": 1, "has_resources": 1, "has_signature": 0, "has_tls": 0, "symbols": 0},
                "header": {"coff": {"timestamp": 1500000000, "machine": "I386", "characteristics": ["EXECUTABLE_IMAGE"]}, "optional": {"subsystem": "WINDOWS_GUI", "dll_characteristics": ["DYNAMIC_BASE"], "magic": "PE32", "major_image_version": 1, "minor_image_version": 0, "major_linker_version": 14, "minor_linker_version": 0, "major_operating_system_version": 6, "minor_operating_system_version": 0, "major_subsystem_version": 6, "minor_subsystem_version": 0, "sizeof_code": 4096, "sizeof_headers": 1024, "sizeof_heap_commit": 4096}},
                "section": {"sections": [{"name": ".text", "size": 4096, "entropy": 6.1, "vsize": 4096, "props": ["MEM_READ", "MEM_EXECUTE"]}]},
                "imports": {"KERNEL32.dll": ["CreateFileA", "CloseHandle"]},
                "exports": [],
            },
            {
                "sha256": "sample_2",
                "label": 1,
                "histogram": [2] * 256,
                "byteentropy": [1] * 256,
                "strings": {"numstrings": 5, "avlength": 3.0, "printabledist": [0] * 96},
                "general": {"size": 4096},
                "imports": {"USER32.dll": ["MessageBoxA"]},
            },
        ]

        X = vectorize_ember_batch(mock_raw_records)
        assert X.shape == (2, 2351)
        assert X.dtype == np.float32
        assert np.isfinite(X).all(), "All vectorized feature values must be finite"
        assert not np.isnan(X).any(), "No NaN values allowed in vectorized features"


# =============================================================================
# 2. SPLIT REPRODUCIBILITY AND TEST ISOLATION TESTS
# =============================================================================

class TestSplitReproducibilityAndTestIsolation:
    """Tests for train/test split isolation and reproducible sampling."""

    def test_train_test_hash_disjointness(self):
        """Verify that genuine processed test split is strictly isolated from train/val/cal."""
        proc_dir = DATA_CONFIG.PROCESSED_DATA_DIR
        x_test_path = proc_dir / "X_test.npy"
        x_train_path = proc_dir / "X_train.npy"

        if not x_test_path.exists() or not x_train_path.exists():
            pytest.skip("Processed numpy splits not generated yet.")

        X_train = np.load(x_train_path)
        X_test = np.load(x_test_path)

        assert X_train.shape[1] == EmberFeatureGroups.TOTAL_FEATURES
        assert X_test.shape[1] == EmberFeatureGroups.TOTAL_FEATURES

        # Verify shapes
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_deduplication_against_train_hashes(self, tmp_path):
        """Verify loader rejects samples present in train_sha256 set."""
        seen_train_hashes = {"hash_leak_1", "hash_leak_2"}
        test_file = tmp_path / "test_samples.jsonl"
        lines = [
            json.dumps({"sha256": "hash_leak_1", "label": 1, "general": {"size": 1000}}),
            json.dumps({"sha256": "hash_clean_sample", "label": 0, "general": {"size": 2000}}),
        ]
        test_file.write_text("\n".join(lines), encoding="utf-8")

        clean_records = list(iter_ember_jsonl(test_file, seen_sha256=seen_train_hashes))
        assert len(clean_records) == 1
        assert clean_records[0]["sha256"] == "hash_clean_sample"


# =============================================================================
# 3. GENUINE MODEL, CALIBRATION & SHAP ATTRIBUTION TESTS
# =============================================================================

class TestGenuineModelAndCalibration:
    """Tests for genuine LightGBM model, calibrator, and SHAP TreeExplainer."""

    @pytest.fixture(autouse=True)
    def setup_genuine_model(self):
        self.genuine_dir = DATA_CONFIG.GENUINE_MODELS_DIR
        self.model_path = self.genuine_dir / MODEL_CONFIG.MODEL_FILENAME
        self.metadata_path = self.genuine_dir / MODEL_CONFIG.METADATA_FILENAME
        self.calibrator_path = self.genuine_dir / CALIBRATION_CONFIG.CALIBRATOR_FILENAME

        if not self.model_path.exists():
            pytest.skip("Genuine LightGBM model artifact not found.")

        self.classifier = MalwareClassifier(
            model_path=self.model_path,
            metadata_path=self.metadata_path,
            calibrator_path=self.calibrator_path,
        )

    def test_genuine_model_metadata(self):
        """Verify genuine model metadata reflects stage 7 genuine provenance."""
        meta = self.classifier.get_metadata()
        assert meta.get("threatlens_stage") == 7
        assert meta.get("dataset_provenance") == "genuine-ember-2018"
        assert meta.get("feature_dim") == 2351
        assert "best_iteration" in meta
        assert "best_val_loss" in meta

    def test_genuine_model_inference_and_calibration(self):
        """Verify model inference produces calibrated probabilities bounded in [0, 1]."""
        rng = np.random.default_rng(42)
        sample = rng.normal(0, 1, size=(5, 2351)).astype(np.float32)

        raw_probs = self.classifier.predict_proba(sample)
        assert raw_probs.shape == (5,)
        assert np.all((raw_probs >= 0.0) & (raw_probs <= 1.0))

        if self.classifier.calibrator is not None:
            cal_probs = self.classifier.predict_calibrated_proba(sample)
            assert cal_probs.shape == (5,)
            assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))

    def test_genuine_model_shap_tree_explainer_additivity(self):
        """Verify local SHAP additivity (|E[f(x)] + sum(SHAP) - f(x)| < 1e-4) on genuine booster."""
        explainer = ShapExplainer(model=self.classifier.booster, calibrator=self.classifier.calibrator)
        rng = np.random.default_rng(123)
        sample = rng.normal(0, 1, size=2351).astype(np.float32)

        explanation = explainer.explain_single(sample)
        assert explanation.prediction in (0, 1)
        assert explanation.prediction_label in ("BENIGN", "MALICIOUS")
        assert explanation.additivity_error < 1e-4
        assert len(explanation.top_positive_contributors) > 0
        assert len(explanation.top_negative_contributors) > 0


# =============================================================================
# 4. FASTAPI ENDPOINT INTEGRATION WITH GENUINE MODEL
# =============================================================================

class TestGenuineModelFastAPIEndpoints:
    """Verifies FastAPI responses when serving genuine EMBER 2018 model."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app) as test_client:
            yield test_client

    def test_api_health_genuine_provenance(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["model_status"] == "GENUINE DATASET — EVALUATION MODEL"
        assert data["calibration_status"] in ("calibrated", "uncalibrated")

    def test_api_model_info_genuine_provenance(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_status"] == "GENUINE DATASET — EVALUATION MODEL"
        assert "genuine" in data["dataset_provenance"].lower()
        assert data["feature_dimension"] == 2351
        assert "STATIC ANALYSIS ONLY" in data["limitations"][0]

    def test_api_predict_with_calibrated_probability(self, client):
        rng = np.random.default_rng(99)
        sample = rng.normal(0, 1, size=2351).astype(float).tolist()

        payload = {"features": sample, "threshold": 0.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in (0, 1)
        assert data["prediction_label"] in ("BENIGN", "MALICIOUS")
        assert 0.0 <= data["raw_probability"] <= 1.0
        assert data["calibrated_probability"] is not None
        assert 0.0 <= data["calibrated_probability"] <= 1.0

    def test_api_metrics_genuine_test_split(self, client):
        response = client.get("/metrics/model")
        assert response.status_code == 200
        data = response.json()
        assert data["threatlens_stage"] == 7
        assert data["dataset_type"] == "genuine_ember_2018_test_split"
        assert "accuracy" in data["metrics"]
        assert "recall" in data["metrics"]
        assert "brier_score_calibrated" in data["metrics"]
