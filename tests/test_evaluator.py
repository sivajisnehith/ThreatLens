"""
Unit Tests for ThreatLens ModelEvaluator and EvaluationMetrics

Tests metric calculations (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR),
confusion matrix reporting, and image/report artifact generation.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from threatlens.config import EmberFeatureGroups, ModelConfig
from threatlens.model.evaluator import EvaluationMetrics, ModelEvaluator
from threatlens.model.trainer import ModelTrainer


@pytest.fixture(scope="module")
def evaluator_fixture(tmp_path_factory):
    """Create a trained model and test splits for evaluation tests."""
    tmp_dir = tmp_path_factory.mktemp("evaluator_test_data")
    rng = np.random.default_rng(42)
    n_features = EmberFeatureGroups.TOTAL_FEATURES  # 2351

    X_train = rng.normal(0, 1, (40, n_features)).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int32)
    X_val = rng.normal(0, 1, (20, n_features)).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(np.int32)
    X_test = rng.normal(0, 1, (30, n_features)).astype(np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.int32)

    config = ModelConfig(NUM_BOOST_ROUND=10, MIN_DATA_IN_LEAF=5)
    trainer = ModelTrainer(config=config, models_dir=tmp_dir)
    result = trainer.train(X_train, y_train, X_val, y_val)

    evaluator = ModelEvaluator(reports_dir=tmp_dir)
    return evaluator, result.booster, X_test, y_test, tmp_dir


class TestModelEvaluator:
    """Test suite for ModelEvaluator."""

    def test_metric_calculations_types_and_bounds(self, evaluator_fixture):
        """Verify metric bounds [0, 1] and mathematical consistency."""
        evaluator, booster, X_test, y_test, _ = evaluator_fixture
        metrics = evaluator.evaluate(booster, X_test, y_test, threshold=0.5)

        assert isinstance(metrics, EvaluationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert 0.0 <= metrics.pr_auc <= 1.0
        assert 0.0 <= metrics.false_positive_rate <= 1.0
        assert 0.0 <= metrics.false_negative_rate <= 1.0

        # Mathematical consistency checks
        # FPR + Specificity = 1.0 (if TN+FP > 0)
        assert abs((metrics.false_positive_rate + metrics.specificity) - 1.0) < 1e-5
        # FNR + Recall = 1.0 (if TP+FN > 0)
        assert abs((metrics.false_negative_rate + metrics.recall) - 1.0) < 1e-5

        # Confusion matrix sum equals sample count
        total = metrics.true_positives + metrics.true_negatives + metrics.false_positives + metrics.false_negatives
        assert total == len(y_test)

    def test_feature_importance_extraction(self, evaluator_fixture):
        """Verify extraction and ranking of feature importances."""
        evaluator, booster, _, _, _ = evaluator_fixture
        imp_data = evaluator.extract_feature_importance(booster, top_n=20)

        assert imp_data["total_features"] == EmberFeatureGroups.TOTAL_FEATURES
        assert len(imp_data["top_by_gain"]) <= 20
        assert imp_data["top_by_gain"][0]["rank"] == 1
        # Ranked descending by gain
        gains = [item["gain_importance"] for item in imp_data["top_by_gain"]]
        assert gains == sorted(gains, reverse=True)

    def test_artifact_generation(self, evaluator_fixture):
        """Verify generation of valid JSON reports and PNG plots."""
        evaluator, booster, X_test, y_test, tmp_dir = evaluator_fixture
        artifacts = evaluator.generate_evaluation_artifacts(booster, X_test, y_test)

        assert artifacts["classification_report"].exists()
        assert artifacts["confusion_matrix_png"].exists()
        assert artifacts["roc_curve_png"].exists()
        assert artifacts["precision_recall_curve_png"].exists()
        assert artifacts["feature_importance_json"].exists()

        # Verify generated PNGs are valid images
        for key in ["confusion_matrix_png", "roc_curve_png", "precision_recall_curve_png"]:
            img_path = artifacts[key]
            assert img_path.stat().st_size > 0
            with Image.open(img_path) as img:
                assert img.format == "PNG"
                assert img.width > 200
                assert img.height > 200
