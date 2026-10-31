"""
ThreatLens Stage 5: Probability Calibration & Threshold Analysis Tests

Tests sigmoid calibration (Platt scaling), isotonic regression, ECE/MCE calculations,
insufficient data and invalid input rejections, artifact save/load, test split isolation,
and threshold trade-off analysis.
"""

from pathlib import Path
import numpy as np
import pytest

from threatlens.config import DATA_CONFIG, EmberFeatureGroups
from threatlens.model.calibration import (
    CalibrationMetrics,
    CalibrationResult,
    CalibratorNotFittedError,
    InsufficientDataError,
    InvalidProbabilityError,
    ProbabilityCalibrator,
    compute_ece,
)
from threatlens.model.classifier import MalwareClassifier
from threatlens.model.threshold_analysis import (
    ThresholdAnalysisResult,
    ThresholdEvaluator,
    ThresholdPoint,
)


@pytest.fixture(scope="module")
def calibration_data():
    """Load calibration split from data/processed."""
    x_path = DATA_CONFIG.PROCESSED_DATA_DIR / "X_cal.npy"
    y_path = DATA_CONFIG.PROCESSED_DATA_DIR / "y_cal.npy"
    if not x_path.exists() or not y_path.exists():
        pytest.skip("Calibration split files not found")

    X_cal = np.load(x_path)
    y_cal = np.load(y_path)
    return X_cal, y_cal


@pytest.fixture(scope="module")
def uncalibrated_probs(calibration_data):
    """Generate uncalibrated probabilities using trained model."""
    X_cal, _ = calibration_data
    clf = MalwareClassifier()
    return clf.predict_proba(X_cal)


class TestProbabilityCalibrator:
    """Tests for ProbabilityCalibrator implementation."""

    def test_sigmoid_calibration_fit_and_predict(self, uncalibrated_probs, calibration_data):
        """Verify Platt scaling (sigmoid) fitting, prediction, and metric computation."""
        _, y_cal = calibration_data
        calibrator = ProbabilityCalibrator(method="sigmoid")

        result = calibrator.fit(uncalibrated_probs, y_cal)

        assert isinstance(result, CalibrationResult)
        assert result.method == "sigmoid"
        assert result.num_samples == len(y_cal)
        assert result.metrics.brier_score_calibrated < result.metrics.brier_score_uncalibrated
        assert calibrator.is_fitted is True

        cal_probs = calibrator.predict_proba(uncalibrated_probs)
        assert len(cal_probs) == len(uncalibrated_probs)
        assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

    def test_isotonic_calibration_fit_and_predict(self, uncalibrated_probs, calibration_data):
        """Verify Isotonic regression fitting and monotonic prediction."""
        _, y_cal = calibration_data
        calibrator = ProbabilityCalibrator(method="isotonic")

        result = calibrator.fit(uncalibrated_probs, y_cal)

        assert isinstance(result, CalibrationResult)
        assert result.method == "isotonic"
        assert calibrator.is_fitted is True

        cal_probs = calibrator.predict_proba(uncalibrated_probs)
        assert len(cal_probs) == len(uncalibrated_probs)
        assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

    def test_unfitted_calibrator_raises(self):
        """Verify calling predict_proba on unfitted calibrator raises CalibratorNotFittedError."""
        calibrator = ProbabilityCalibrator()
        with pytest.raises(CalibratorNotFittedError):
            calibrator.predict_proba([0.5, 0.6])

    def test_insufficient_samples_raises(self):
        """Verify fitting on fewer than 20 samples raises InsufficientDataError."""
        calibrator = ProbabilityCalibrator()
        probs = np.array([0.4, 0.6, 0.5], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.int32)

        with pytest.raises(InsufficientDataError):
            calibrator.fit(probs, y)

    def test_single_class_raises(self):
        """Verify fitting on data containing only one class raises InsufficientDataError."""
        calibrator = ProbabilityCalibrator()
        probs = np.full(30, 0.6, dtype=np.float32)
        y = np.ones(30, dtype=np.int32)

        with pytest.raises(InsufficientDataError):
            calibrator.fit(probs, y)

    def test_invalid_probability_range_raises(self):
        """Verify probabilities outside [0, 1] raise InvalidProbabilityError."""
        calibrator = ProbabilityCalibrator()
        # Margin log-odds instead of probability
        with pytest.raises(InvalidProbabilityError):
            calibrator.validate_probabilities([-1.5, 2.3])

    def test_nan_or_inf_probability_raises(self):
        """Verify NaN or Inf probabilities raise InvalidProbabilityError."""
        calibrator = ProbabilityCalibrator()
        with pytest.raises(InvalidProbabilityError):
            calibrator.validate_probabilities([0.5, np.nan, 0.7])

        with pytest.raises(InvalidProbabilityError):
            calibrator.validate_probabilities([0.5, np.inf, 0.7])

    def test_save_and_load_artifact(self, uncalibrated_probs, calibration_data, tmp_path):
        """Verify serialization and deserialization of fitted calibrator."""
        _, y_cal = calibration_data
        calibrator = ProbabilityCalibrator(method="sigmoid", models_dir=tmp_path)
        calibrator.fit(uncalibrated_probs, y_cal)
        cal_path, meta_path = calibrator.save(output_dir=tmp_path)

        assert cal_path.exists()
        assert meta_path.exists()

        # Load into new instance
        new_cal = ProbabilityCalibrator(models_dir=tmp_path)
        new_cal.load(calibrator_path=cal_path, metadata_path=meta_path)
        assert new_cal.is_fitted is True

        # Check prediction consistency
        orig_preds = calibrator.predict_proba(uncalibrated_probs[:10])
        loaded_preds = new_cal.predict_proba(uncalibrated_probs[:10])
        assert np.allclose(orig_preds, loaded_preds)

    def test_calibration_preserves_uncalibrated_split_isolation(self, calibration_data):
        """Verify calibration never fits or touches the untouched test split."""
        X_cal, y_cal = calibration_data
        X_test = np.load(DATA_CONFIG.PROCESSED_DATA_DIR / "X_test.npy")
        y_test = np.load(DATA_CONFIG.PROCESSED_DATA_DIR / "y_test.npy")

        clf = MalwareClassifier()
        probs_cal = clf.predict_proba(X_cal)
        probs_test = clf.predict_proba(X_test)

        # Fit exclusively on X_cal
        calibrator = ProbabilityCalibrator()
        calibrator.fit(probs_cal, y_cal)

        # Evaluate on test set
        cal_test = calibrator.predict_proba(probs_test)
        assert len(cal_test) == len(X_test)
        # Ensure test probabilities changed and remained bounded
        assert not np.array_equal(cal_test, probs_test)
        assert (cal_test >= 0.0).all() and (cal_test <= 1.0).all()

    def test_ece_computation(self):
        """Verify Expected Calibration Error and Maximum Calibration Error math."""
        # Perfectly calibrated synthetic case
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_prob = np.array([0.05, 0.05, 0.95, 0.95], dtype=float)

        ece, mce = compute_ece(y_true, y_prob, n_bins=10)
        assert 0.0 <= ece <= 0.1
        assert 0.0 <= mce <= 0.1


class TestThresholdAnalysis:
    """Tests for ThresholdEvaluator and operational posture recommendations."""

    def test_evaluate_thresholds_metrics_consistency(self, uncalibrated_probs, calibration_data):
        """Verify confusion matrix and rate identities across multiple thresholds."""
        _, y_cal = calibration_data
        evaluator = ThresholdEvaluator(default_threshold=0.50)

        result = evaluator.evaluate(uncalibrated_probs, y_cal)

        assert isinstance(result, ThresholdAnalysisResult)
        assert result.total_samples == len(y_cal)
        assert len(result.points) > 10

        for p in result.points:
            # Identity 1: TP + FP + TN + FN == Total
            assert p.true_positives + p.false_positives + p.true_negatives + p.false_negatives == len(y_cal)
            # Identity 2: Recall + FNR == 1.0
            assert abs((p.recall + p.false_negative_rate) - 1.0) < 1e-3
            # Identity 3: Specificity + FPR == 1.0
            assert abs((p.specificity + p.false_positive_rate) - 1.0) < 1e-3

    def test_edge_thresholds(self, uncalibrated_probs, calibration_data):
        """Verify extreme edge thresholds: t=0.0 and t=1.0."""
        _, y_cal = calibration_data
        evaluator = ThresholdEvaluator()

        result = evaluator.evaluate(uncalibrated_probs, y_cal, thresholds=[0.0, 1.0])
        p_zero = result.points[0]
        p_one = result.points[1]

        # At threshold 0.0, everything is predicted positive
        assert p_zero.threshold == 0.0
        assert p_zero.true_negatives == 0
        assert p_zero.false_negatives == 0
        assert p_zero.recall == 1.0
        assert p_zero.false_positive_rate == 1.0

        # At threshold 1.0, everything is predicted negative (since probs < 1.0)
        assert p_one.threshold == 1.0
        assert p_one.true_positives == 0
        assert p_one.false_positives == 0
        assert p_one.specificity == 1.0
        assert p_one.false_negative_rate == 1.0

    def test_operational_postures_generated(self, uncalibrated_probs, calibration_data):
        """Verify operational postures (high security vs low interruption vs balanced)."""
        _, y_cal = calibration_data
        evaluator = ThresholdEvaluator()
        result = evaluator.evaluate(uncalibrated_probs, y_cal)

        assert "balanced_f1" in result.postures
        assert "high_security_low_fnr" in result.postures
        assert "low_interruption_low_fpr" in result.postures

        # High security threshold must be <= low interruption threshold
        high_sec_t = result.postures["high_security_low_fnr"].threshold
        low_int_t = result.postures["low_interruption_low_fpr"].threshold
        assert high_sec_t <= low_int_t

    def test_generate_full_report_and_plot(self, uncalibrated_probs, calibration_data, tmp_path):
        """Verify JSON and PNG generation for threshold analysis."""
        _, y_cal = calibration_data
        evaluator = ThresholdEvaluator(reports_dir=tmp_path)
        artifacts = evaluator.generate_full_report(uncalibrated_probs, y_cal, output_dir=tmp_path)

        assert artifacts["json_report"].exists()
        assert artifacts["plot"].exists()
        assert artifacts["json_report"].stat().st_size > 0
        assert artifacts["plot"].stat().st_size > 0
