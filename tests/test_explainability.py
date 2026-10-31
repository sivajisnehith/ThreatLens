"""
ThreatLens Stage 4: SHAP Explainability Engine Test Suite

Tests SHAP TreeExplainer initialization, single and batch explanation mechanics,
feature name mapping, positive/negative direction interpretation, ranking logic,
dimension normalization, exact log-odds additivity, model compatibility,
invalid feature rejections, and JSON serialization.
"""

import json
from pathlib import Path
import tempfile
import pytest
import numpy as np
import lightgbm as lgb

from threatlens.config import DATA_CONFIG, FEATURE_GROUPS, MODEL_CONFIG, EmberFeatureGroups
from threatlens.core.adapter import IncompatibleFeatureError
from threatlens.explainability.explainer import (
    DIRECTION_DECREASES_MALWARE,
    DIRECTION_INCREASES_MALWARE,
    DIRECTION_NEUTRAL,
    BatchExplanation,
    ExplainerModelError,
    ExplainerNotInitializedError,
    ShapExplainer,
    SingleExplanation,
)
from threatlens.explainability.feature_interpreter import (
    FeatureContribution,
    FeatureInterpreter,
)
from threatlens.explainability.global_explanations import (
    GlobalExplainer,
    GlobalImportanceResult,
)
from threatlens.model.classifier import FeatureDimensionError, MalwareClassifier
from threatlens.model.trainer import generate_ember_feature_names


@pytest.fixture(scope="module")
def deterministic_toy_model():
    """Create a fast, deterministic LightGBM model for testing."""
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = EmberFeatureGroups.TOTAL_FEATURES  # 2351
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = (X[:, 1] + X[:, 317] - X[:, 770] > 0).astype(int)

    feature_names = generate_ember_feature_names()
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=False)
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_data_in_leaf": 2,
        "verbosity": -1,
        "random_state": 42,
    }
    booster = lgb.train(params, dtrain, num_boost_round=10)
    return booster, X, y, feature_names


@pytest.fixture(scope="module")
def standard_explainer():
    """Initialize ShapExplainer using the serialized Stage 3 model."""
    model_path = DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME
    if not model_path.exists():
        pytest.skip("Serialized Stage 3 model not found")
    return ShapExplainer(model=model_path)


class TestShapInitialization:
    """Requirement 1 & 10: SHAP initialization and model compatibility."""

    def test_init_with_default_model(self, standard_explainer):
        """Test initializing with existing default serialized model."""
        assert standard_explainer.booster is not None
        assert standard_explainer.explainer is not None
        assert standard_explainer.expected_dim == 2351
        assert len(standard_explainer.feature_names) == 2351
        assert isinstance(standard_explainer.expected_value, float)

    def test_init_with_booster_instance(self, deterministic_toy_model):
        """Test initializing directly with a Booster instance."""
        booster, _, _, _ = deterministic_toy_model
        explainer = ShapExplainer(model=booster)
        assert explainer.booster is booster
        assert explainer.explainer is not None

    def test_init_with_malware_classifier_instance(self):
        """Test initializing with a MalwareClassifier object."""
        model_path = DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME
        if not model_path.exists():
            pytest.skip("Stage 3 model not found")
        classifier = MalwareClassifier(model_path=model_path)
        explainer = ShapExplainer(model=classifier)
        assert explainer.booster is classifier.booster

    def test_init_with_missing_model_raises(self):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ShapExplainer(model="nonexistent_model_file.txt")

    def test_init_with_corrupted_model_file_raises(self, tmp_path):
        """Test that invalid/corrupt model file raises ExplainerModelError."""
        corrupt_file = tmp_path / "corrupt_model.txt"
        corrupt_file.write_text("invalid lightgbm model content")
        with pytest.raises(ExplainerModelError):
            ShapExplainer(model=corrupt_file)


class TestSinglePredictionExplanation:
    """Requirement 2: Single prediction explanation."""

    def test_explain_single_structure(self, standard_explainer):
        """Verify the structured response of explain_single."""
        rng = np.random.default_rng(101)
        sample = rng.normal(0, 1, size=(2351,)).astype(np.float32)

        exp = standard_explainer.explain_single(sample, top_k=15, threshold=0.5)

        assert isinstance(exp, SingleExplanation)
        assert exp.prediction in (0, 1)
        assert exp.prediction_label in ("BENIGN", "MALICIOUS")
        assert 0.0 <= exp.probability <= 1.0
        assert 0.0 <= exp.benign_probability <= 1.0
        assert abs((exp.probability + exp.benign_probability) - 1.0) < 1e-5
        assert isinstance(exp.raw_score, float)
        assert isinstance(exp.expected_value, float)
        assert len(exp.feature_values) == 2351
        assert len(exp.shap_values) == 2351
        assert len(exp.feature_names) == 2351
        assert len(exp.top_contributors) <= 15
        assert len(exp.top_positive_contributors) <= 15
        assert len(exp.top_negative_contributors) <= 15
        assert exp.is_synthetic_model is True
        assert "synthetic" in exp.disclaimer.lower()

    def test_explain_single_from_2d_row(self, standard_explainer):
        """Verify explain_single handles (1, 2351) inputs correctly."""
        sample = np.zeros((1, 2351), dtype=np.float32)
        exp = standard_explainer.explain_single(sample)
        assert isinstance(exp, SingleExplanation)
        assert len(exp.feature_values) == 2351


class TestBatchExplanations:
    """Requirement 3: Batch explanations."""

    def test_explain_batch_structure(self, standard_explainer):
        """Verify batch explanation execution and dimension consistency."""
        rng = np.random.default_rng(102)
        batch = rng.normal(0, 1, size=(5, 2351)).astype(np.float32)

        batch_exp = standard_explainer.explain_batch(batch, top_k=10)

        assert isinstance(batch_exp, BatchExplanation)
        assert batch_exp.count == 5
        assert len(batch_exp.predictions) == 5
        assert len(batch_exp.probabilities) == 5
        assert len(batch_exp.raw_scores) == 5
        assert batch_exp.shap_matrix.shape == (5, 2351)
        assert batch_exp.feature_matrix.shape == (5, 2351)
        assert len(batch_exp.mean_abs_shap) == 2351

        # Test extracting single explanation from batch
        single_exp = batch_exp.get_explanation(index=2, interpreter=standard_explainer.interpreter)
        assert isinstance(single_exp, SingleExplanation)
        assert single_exp.prediction == batch_exp.predictions[2]
        assert abs(single_exp.probability - batch_exp.probabilities[2]) < 1e-6

    def test_batch_explanation_out_of_bounds_raises(self, standard_explainer):
        """Verify get_explanation raises IndexError when out of bounds."""
        batch = np.zeros((3, 2351), dtype=np.float32)
        batch_exp = standard_explainer.explain_batch(batch)
        with pytest.raises(IndexError):
            batch_exp.get_explanation(5, interpreter=standard_explainer.interpreter)


class TestFeatureNameMapping:
    """Requirement 4 & 11: Feature name mapping and missing names fallback."""

    def test_canonical_feature_names_mapping(self, standard_explainer):
        """Verify all 2,351 features match EMBER groups and canonical names."""
        interpreter = standard_explainer.interpreter
        assert interpreter is not None

        # Check group boundaries
        for group in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(group)
            for idx in [start, (start + end) // 2, end - 1]:
                assert interpreter.get_feature_group(idx) == group
                assert group in interpreter.get_feature_name(idx)

    def test_missing_feature_names_fallback(self):
        """Test fallback when model does not specify feature names."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(20, 2351)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        dtrain = lgb.Dataset(X, label=y)  # No feature names provided
        booster = lgb.train({"objective": "binary", "verbosity": -1}, dtrain, num_boost_round=3)

        explainer = ShapExplainer(model=booster)
        assert len(explainer.feature_names) == 2351
        assert explainer.feature_names[0].startswith("ember_")


class TestDirectionalInterpretation:
    """Requirement 5 & 6: Positive and negative contribution interpretation."""

    def test_positive_and_negative_contributions(self, deterministic_toy_model):
        """Verify positive SHAP represents movement toward malicious class (1)."""
        booster, X, y, _ = deterministic_toy_model
        explainer = ShapExplainer(model=booster)

        exp = explainer.explain_single(X[0])

        for pos in exp.top_positive_contributors:
            assert pos.shap_value > 0
            assert pos.direction == DIRECTION_INCREASES_MALWARE

        for neg in exp.top_negative_contributors:
            assert neg.shap_value < 0
            assert neg.direction == DIRECTION_DECREASES_MALWARE

    def test_interpreter_direction_values(self):
        """Test direction constants on edge values."""
        interpreter = FeatureInterpreter()
        contrib_pos = interpreter.interpret_contribution(1, feature_value=0.5, shap_value=0.1)
        contrib_neg = interpreter.interpret_contribution(1, feature_value=0.5, shap_value=-0.1)
        contrib_zero = interpreter.interpret_contribution(1, feature_value=0.5, shap_value=0.0)

        assert contrib_pos.direction == DIRECTION_INCREASES_MALWARE
        assert contrib_neg.direction == DIRECTION_DECREASES_MALWARE
        assert contrib_zero.direction == DIRECTION_NEUTRAL


class TestFeatureRanking:
    """Requirement 7: Feature ranking by absolute contribution."""

    def test_features_ranked_by_absolute_contribution(self, standard_explainer):
        """Verify that top_contributors are strictly ordered by |SHAP| descending."""
        rng = np.random.default_rng(200)
        sample = rng.normal(0, 1, size=(2351,)).astype(np.float32)

        exp = standard_explainer.explain_single(sample, top_k=20)
        abs_vals = [c.abs_shap_value for c in exp.top_contributors]

        for i in range(len(abs_vals) - 1):
            assert abs_vals[i] >= abs_vals[i + 1], "Features are not sorted by absolute SHAP descending"


class TestDimensionNormalization:
    """Requirement 8: SHAP output dimensions and normalization."""

    def test_normalize_shap_output_various_shapes(self, standard_explainer):
        """Verify _normalize_shap_output converts various raw SHAP formats to (N, 2351)."""
        n_samples = 4
        n_features = 2351

        # Case 1: 2D array (N, 2351)
        raw_2d = np.ones((n_samples, n_features), dtype=np.float32)
        norm_2d = standard_explainer._normalize_shap_output(raw_2d, n_samples)
        assert norm_2d.shape == (n_samples, n_features)

        # Case 2: List of two arrays [class0, class1]
        raw_list = [np.zeros((n_samples, n_features)), np.ones((n_samples, n_features))]
        norm_list = standard_explainer._normalize_shap_output(raw_list, n_samples)
        assert norm_list.shape == (n_samples, n_features)
        assert np.allclose(norm_list, 1.0)

        # Case 3: 3D array (N, 2351, 2)
        raw_3d = np.zeros((n_samples, n_features, 2), dtype=np.float32)
        raw_3d[:, :, 1] = 2.0
        norm_3d = standard_explainer._normalize_shap_output(raw_3d, n_samples)
        assert norm_3d.shape == (n_samples, n_features)
        assert np.allclose(norm_3d, 2.0)

        # Case 4: 1D single sample (2351,)
        raw_1d = np.ones(n_features, dtype=np.float32)
        norm_1d = standard_explainer._normalize_shap_output(raw_1d, 1)
        assert norm_1d.shape == (1, n_features)


class TestAdditivity:
    """Requirement 9: Additivity checks for TreeExplainer on log-odds margin."""

    def test_exact_additivity_property(self, standard_explainer):
        """Verify baseline + sum(SHAP) == raw_score within numerical tolerance."""
        rng = np.random.default_rng(300)
        X_test = rng.normal(0, 1, size=(5, 2351)).astype(np.float32)

        for i in range(len(X_test)):
            exp = standard_explainer.explain_single(X_test[i])
            expected_sum = exp.expected_value + np.sum(exp.shap_values)
            assert abs(expected_sum - exp.raw_score) < 1e-4
            assert exp.additivity_error < 1e-4


class TestFeatureValidationAndRejection:
    """Requirement 12: Invalid feature vector and schema isolation tests."""

    def test_strictly_rejects_custom_pe_schema_dictionary(self, standard_explainer):
        """Verify strict isolation against custom PE feature extractor dictionary."""
        custom_pe_dict = {
            "schema_version": "1.0.0",
            "file_size": 10240,
            "dos_header": {"e_magic": 23117},
            "sections": [],
        }
        with pytest.raises(IncompatibleFeatureError):
            standard_explainer.explain_single(custom_pe_dict)

    def test_rejects_generic_dictionary(self, standard_explainer):
        """Verify rejection of generic dicts."""
        with pytest.raises(IncompatibleFeatureError):
            standard_explainer.explain_single({"feature1": 0.5})

    def test_rejects_wrong_feature_dimension(self, standard_explainer):
        """Verify rejection of 85-dimensional or wrong dimension arrays."""
        # Custom PE extractor produces 85 dimensions
        wrong_85 = np.zeros(85, dtype=np.float32)
        with pytest.raises(FeatureDimensionError):
            standard_explainer.explain_single(wrong_85)

        wrong_2d = np.zeros((2, 100), dtype=np.float32)
        with pytest.raises(FeatureDimensionError):
            standard_explainer.explain_single(wrong_2d)

    def test_rejects_infinite_values(self, standard_explainer):
        """Verify rejection of infinite feature values."""
        sample = np.zeros(2351, dtype=np.float32)
        sample[10] = np.inf
        with pytest.raises(ValueError):
            standard_explainer.explain_single(sample)


class TestJsonSerialization:
    """Requirement 13: Explanation serialization to JSON."""

    def test_single_explanation_json_serialization(self, standard_explainer, tmp_path):
        """Verify SingleExplanation serializes to dictionary and JSON file cleanly."""
        sample = np.zeros(2351, dtype=np.float32)
        exp = standard_explainer.explain_single(sample)

        exp_dict = exp.to_dict()
        assert isinstance(exp_dict, dict)
        assert isinstance(exp_dict["prediction"], int)
        assert isinstance(exp_dict["probability"], float)

        # Ensure valid JSON encoding
        json_str = exp.to_json()
        decoded = json.loads(json_str)
        assert decoded["prediction"] == exp.prediction

        # Ensure writing to file succeeds
        out_file = tmp_path / "single_explanation.json"
        exp.to_json(out_file)
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_batch_explanation_json_serialization(self, standard_explainer):
        """Verify BatchExplanation serializes to dictionary cleanly."""
        samples = np.zeros((3, 2351), dtype=np.float32)
        batch_exp = standard_explainer.explain_batch(samples)
        data = batch_exp.to_dict()
        assert isinstance(data, dict)
        assert data["count"] == 3
        # JSON dump check
        json_str = json.dumps(data)
        assert len(json_str) > 0


class TestGlobalExplainerAndReporting:
    """Test GlobalExplainer analysis and artifact generation."""

    def test_calculate_global_importance(self, standard_explainer):
        """Test global importance computation and group aggregations."""
        rng = np.random.default_rng(400)
        X = rng.normal(0, 1, size=(20, 2351)).astype(np.float32)

        global_engine = GlobalExplainer(explainer=standard_explainer)
        result, X_eval, shap_matrix = global_engine.calculate_global_importance(X, top_k=10)

        assert isinstance(result, GlobalImportanceResult)
        assert result.total_samples == 20
        assert len(result.top_features_by_shap) == 10
        assert len(result.group_importance) == 8
        assert "histogram" in result.group_importance
        assert "imports" in result.group_importance
        assert shap_matrix.shape == (20, 2351)

    def test_generate_full_report_artifacts(self, standard_explainer, tmp_path):
        """Verify that generate_full_report creates all 4 required artifacts."""
        rng = np.random.default_rng(500)
        X = rng.normal(0, 1, size=(10, 2351)).astype(np.float32)

        global_engine = GlobalExplainer(explainer=standard_explainer, reports_dir=tmp_path)
        artifacts = global_engine.generate_full_report(X, output_dir=tmp_path, top_k=5)

        assert artifacts["json_report"].exists()
        assert artifacts["summary_plot"].exists()
        assert artifacts["bar_plot"].exists()
        assert artifacts["group_plot"].exists()

        assert artifacts["json_report"].stat().st_size > 0
        assert artifacts["summary_plot"].stat().st_size > 0
        assert artifacts["bar_plot"].stat().st_size > 0
        assert artifacts["group_plot"].stat().st_size > 0
