"""
Unit Tests for ThreatLens MalwareClassifier Interface

Tests model loading, single and batch inference, dimension checking,
schema isolation (rejection of custom PE features), and prediction reproducibility.
"""

from pathlib import Path
import numpy as np
import pytest

from threatlens.config import DATA_CONFIG, MODEL_CONFIG, EmberFeatureGroups, ModelConfig
from threatlens.core.adapter import IncompatibleFeatureError
from threatlens.model.classifier import (
    BatchPredictionResult,
    FeatureDimensionError,
    MalwareClassifier,
    ModelNotLoadedError,
    PredictionResult,
)
from threatlens.model.trainer import ModelTrainer


@pytest.fixture(scope="module")
def trained_test_model(tmp_path_factory):
    """Create a temporary trained LightGBM model for testing inference."""
    tmp_dir = tmp_path_factory.mktemp("classifier_test_model")
    rng = np.random.default_rng(42)
    n_features = EmberFeatureGroups.TOTAL_FEATURES  # 2351

    X_train = rng.normal(0, 1, (40, n_features)).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int32)
    X_val = rng.normal(0, 1, (20, n_features)).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(np.int32)

    config = ModelConfig(NUM_BOOST_ROUND=10, MIN_DATA_IN_LEAF=5)
    trainer = ModelTrainer(config=config, models_dir=tmp_dir)
    result = trainer.train(X_train, y_train, X_val, y_val)
    model_path, meta_path = trainer.save_model(result, output_dir=tmp_dir)

    return model_path, meta_path


class TestMalwareClassifier:
    """Test suite for MalwareClassifier."""

    def test_model_loading_success(self, trained_test_model):
        """Verify model and metadata load properly."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)
        assert clf.is_loaded is True
        assert clf.booster is not None
        assert clf.expected_dim == 2351
        assert len(clf.get_feature_names()) == 2351

    def test_missing_model_file_raises(self, tmp_path):
        """Verify FileNotFoundError when model path does not exist."""
        fake_path = tmp_path / "non_existent_model.txt"
        with pytest.raises(FileNotFoundError):
            MalwareClassifier(model_path=fake_path)

    def test_single_prediction(self, trained_test_model):
        """Verify predict_single produces valid structured PredictionResult."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        x_single = np.zeros(2351, dtype=np.float32)
        res = clf.predict_single(x_single, threshold=0.5)

        assert isinstance(res, PredictionResult)
        assert res.label in [0, 1]
        assert res.label_name in ["BENIGN", "MALICIOUS"]
        assert 0.0 <= res.probability <= 1.0
        assert 0.0 <= res.benign_probability <= 1.0
        assert abs(res.probability + res.benign_probability - 1.0) < 1e-5
        assert res.feature_dim == 2351

    def test_batch_prediction(self, trained_test_model):
        """Verify predict_batch produces valid BatchPredictionResult."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        X_batch = np.random.randn(5, 2351).astype(np.float32)
        batch_res = clf.predict_batch(X_batch, threshold=0.5)

        assert isinstance(batch_res, BatchPredictionResult)
        assert len(batch_res.labels) == 5
        assert len(batch_res.probabilities) == 5
        assert batch_res.count == 5
        assert ((batch_res.probabilities >= 0.0) & (batch_res.probabilities <= 1.0)).all()

    def test_feature_dimension_mismatch_raises(self, trained_test_model):
        """Verify FeatureDimensionError when feature vector is not 2,351."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        # 1D wrong dimension (85 features from custom PE extractor)
        x_wrong_1d = np.zeros(85, dtype=np.float32)
        with pytest.raises(FeatureDimensionError) as exc_info:
            clf.predict_single(x_wrong_1d)
        assert "expected 2351" in str(exc_info.value)

        # 2D wrong dimension
        X_wrong_2d = np.zeros((3, 100), dtype=np.float32)
        with pytest.raises(FeatureDimensionError) as exc_info:
            clf.predict_batch(X_wrong_2d)
        assert "expected 2351" in str(exc_info.value)

    def test_strictly_rejects_custom_pe_schema_dictionary(self, trained_test_model):
        """
        CRITICAL ARCHITECTURAL SAFETY TEST:
        Verify classifier strictly raises IncompatibleFeatureError when given
        a custom PE feature dictionary.
        """
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        custom_pe_dict = {
            "schema_version": "1.0.0",
            "dos_header": {"e_magic": 23117},
            "sections": [],
        }

        with pytest.raises(IncompatibleFeatureError) as exc_info:
            clf.predict_single(custom_pe_dict)  # type: ignore
        assert "Incompatible feature schema" in str(exc_info.value)
        assert "strictly prohibited" in str(exc_info.value)

    def test_rejects_infinite_values(self, trained_test_model):
        """Verify ValueError is raised if input contains infinite values."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        x_inf = np.zeros(2351, dtype=np.float32)
        x_inf[10] = np.inf

        with pytest.raises(ValueError) as exc_info:
            clf.predict_proba(x_inf)
        assert "infinite values" in str(exc_info.value).lower()

    def test_prediction_determinism(self, trained_test_model):
        """Verify identical inputs yield bitwise identical probabilities."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        x = np.random.randn(2351).astype(np.float32)
        p1 = clf.predict_proba(x)
        p2 = clf.predict_proba(x)

        assert np.array_equal(p1, p2)

    def test_threshold_adjustment(self, trained_test_model):
        """Verify changing decision threshold alters predictions as expected."""
        model_path, meta_path = trained_test_model
        clf = MalwareClassifier(model_path=model_path, metadata_path=meta_path)

        x = np.random.randn(2351).astype(np.float32)
        prob = float(clf.predict_proba(x)[0])

        # If threshold is below probability, label must be 1; if above, 0
        res_low = clf.predict_single(x, threshold=prob - 0.05 if prob > 0.1 else 0.0)
        assert res_low.label == 1

        res_high = clf.predict_single(x, threshold=prob + 0.05 if prob < 0.9 else 1.0)
        assert res_high.label == 0
