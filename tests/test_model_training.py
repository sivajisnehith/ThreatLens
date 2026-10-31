"""
Unit Tests for ThreatLens ModelTrainer

Tests initialization, LightGBM training on 2,351-dimensional EMBER features,
early stopping, serialization, dimension enforcement, and label validation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from threatlens.config import EmberFeatureGroups, ModelConfig
from threatlens.model.trainer import (
    ModelTrainer,
    TrainingResult,
    generate_ember_feature_names,
)


@pytest.fixture
def synthetic_ember_splits():
    """Generate small synthetic datasets conforming strictly to the 2,351 EMBER dimension."""
    rng = np.random.default_rng(42)
    n_features = EmberFeatureGroups.TOTAL_FEATURES  # 2351

    n_train = 60
    X_train = rng.normal(0, 1, (n_train, n_features)).astype(np.float32)
    # Inject signal into first 5 features so loss improves and early stopping can evaluate
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int32)

    n_val = 20
    X_val = rng.normal(0, 1, (n_val, n_features)).astype(np.float32)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int32)

    return X_train, y_train, X_val, y_val


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_initialization_defaults(self):
        """Verify trainer initialization with default config."""
        trainer = ModelTrainer()
        assert trainer.config.OBJECTIVE == "binary"
        assert trainer.config.NUM_LEAVES == 31
        assert trainer.models_dir.exists()

    def test_feature_names_generator(self):
        """Verify generated EMBER feature names match exact 2,351 count."""
        names = generate_ember_feature_names()
        assert len(names) == EmberFeatureGroups.TOTAL_FEATURES
        assert names[0].startswith("ember_histogram_")
        assert names[-1].startswith("ember_exports_")

    def test_training_pipeline_success(self, synthetic_ember_splits):
        """Verify end-to-end training execution on 2,351-dimensional features."""
        X_train, y_train, X_val, y_val = synthetic_ember_splits

        config = ModelConfig(
            NUM_BOOST_ROUND=20,
            EARLY_STOPPING_ROUNDS=5,
            LEARNING_RATE=0.1,
            MIN_DATA_IN_LEAF=5,
        )
        trainer = ModelTrainer(config=config)
        result = trainer.train(X_train, y_train, X_val, y_val)

        assert isinstance(result, TrainingResult)
        assert result.booster is not None
        assert result.best_iteration >= 1
        assert result.train_samples == len(X_train)
        assert result.val_samples == len(X_val)
        assert len(result.feature_names) == EmberFeatureGroups.TOTAL_FEATURES
        assert result.train_duration_sec > 0

    def test_serialization_and_metadata(self, synthetic_ember_splits, tmp_path):
        """Verify model saving to lightgbm_ember.txt and model_metadata.json."""
        X_train, y_train, X_val, y_val = synthetic_ember_splits
        config = ModelConfig(NUM_BOOST_ROUND=10, MIN_DATA_IN_LEAF=5)
        trainer = ModelTrainer(config=config, models_dir=tmp_path)
        result = trainer.train(X_train, y_train, X_val, y_val)

        model_path, meta_path = trainer.save_model(result, output_dir=tmp_path)

        assert model_path.exists()
        assert model_path.stat().st_size > 0
        assert meta_path.exists()

        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["threatlens_stage"] == 3
        assert meta["feature_dim"] == EmberFeatureGroups.TOTAL_FEATURES
        assert meta["model_format"] == "text"
        assert meta["best_iteration"] == result.best_iteration

    def test_rejects_wrong_feature_dimension(self, synthetic_ember_splits):
        """Verify trainer raises ValueError when feature dimension is not 2,351."""
        X_train, y_train, X_val, y_val = synthetic_ember_splits
        # Slice to wrong dimension (e.g. 100 features)
        X_train_wrong = X_train[:, :100]

        trainer = ModelTrainer()
        with pytest.raises(ValueError) as exc_info:
            trainer.train(X_train_wrong, y_train, X_val, y_val)
        assert "dimension mismatch" in str(exc_info.value).lower()

    def test_rejects_invalid_labels(self, synthetic_ember_splits):
        """Verify trainer raises ValueError when labels contain non-binary values."""
        X_train, y_train, X_val, y_val = synthetic_ember_splits
        y_corrupt = y_train.copy()
        y_corrupt[0] = 5  # Invalid label

        trainer = ModelTrainer()
        with pytest.raises(ValueError) as exc_info:
            trainer.train(X_train, y_corrupt, X_val, y_val)
        assert "invalid labels" in str(exc_info.value).lower()
