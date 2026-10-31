"""
ThreatLens Probability Calibration Engine (Stage 5)

Implements post-hoc probability calibration for the primary LightGBM binary classifier
using the dedicated calibration split (X_cal, y_cal). Supports Platt scaling (sigmoid calibration)
and Isotonic regression, calculates Brier score, Log Loss, Expected Calibration Error (ECE),
and Maximum Calibration Error (MCE), and serializes calibration artifacts.

CRITICAL DATASET NOTICE:
Current models were trained on synthetic EMBER representations for infrastructure validation.
Calibrated probabilities on synthetic data must be treated as experimental demonstrations and
must NOT be interpreted as real-world malware likelihood.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import _SigmoidCalibration, IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from threatlens.config import CALIBRATION_CONFIG, DATA_CONFIG, MODEL_CONFIG
from threatlens.utils.logger import get_logger

logger = get_logger("calibration")

CALIBRATION_SYNTHETIC_DISCLAIMER: str = (
    "EXPERIMENTAL: Probability calibrator was fitted on synthetic data distributions. "
    "Calibrated probabilities demonstrate pipeline calibration mechanics and must NOT be "
    "interpreted as real-world malware probability."
)


class CalibrationError(Exception):
    """Base exception for calibration failures."""
    pass


class CalibratorNotFittedError(CalibrationError):
    """Raised when calibration inference is attempted before fitting or loading."""
    pass


class InsufficientDataError(CalibrationError):
    """Raised when the calibration dataset has insufficient samples or single-class data."""
    pass


class InvalidProbabilityError(CalibrationError):
    """Raised when input probabilities violate the [0, 1] range or contain non-finite numbers."""
    pass


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary ground-truth labels (0 or 1)
        y_prob: Predicted probabilities in [0.0, 1.0]
        n_bins: Number of equal-width probability bins

    Returns:
        Tuple of (ece, mce)
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    if n == 0:
        return 0.0, 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (count / n) * diff
            mce = max(mce, diff)

    return float(ece), float(mce)


@dataclass
class CalibrationMetrics:
    """Metrics comparing uncalibrated vs calibrated model performance."""
    brier_score_uncalibrated: float
    brier_score_calibrated: float
    log_loss_uncalibrated: float
    log_loss_calibrated: float
    ece_uncalibrated: float
    ece_calibrated: float
    mce_uncalibrated: float
    mce_calibrated: float
    brier_improvement: float
    log_loss_improvement: float
    ece_improvement: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class CalibrationResult:
    """Container for calibration results and metadata."""
    method: str
    num_samples: int
    metrics: CalibrationMetrics
    calibrator_file: str
    metadata_file: str
    timestamp: str
    is_synthetic: bool = True
    disclaimer: str = CALIBRATION_SYNTHETIC_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "method": self.method,
            "num_samples": self.num_samples,
            "metrics": self.metrics.to_dict(),
            "calibrator_file": self.calibrator_file,
            "metadata_file": self.metadata_file,
            "timestamp": self.timestamp,
            "is_synthetic": self.is_synthetic,
            "disclaimer": self.disclaimer,
        }


class ProbabilityCalibrator:
    """
    Probability calibration engine for ThreatLens malware classification.
    Supports Platt scaling ('sigmoid') and Isotonic regression ('isotonic').
    """

    def __init__(
        self,
        method: str = "sigmoid",
        models_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize probability calibrator.

        Args:
            method: Calibration method ('sigmoid' for Platt scaling or 'isotonic')
            models_dir: Directory containing or destination for calibration artifacts
        """
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unsupported calibration method: '{method}'. Choose 'sigmoid' or 'isotonic'.")

        self.method = method
        self.models_dir = Path(models_dir) if models_dir else DATA_CONFIG.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.calibrator_filename = CALIBRATION_CONFIG.CALIBRATOR_FILENAME
        self.metadata_filename = CALIBRATION_CONFIG.METADATA_FILENAME

        self.calibrator: Optional[Union[_SigmoidCalibration, IsotonicRegression]] = None
        self.metadata: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def validate_probabilities(self, probs: Any) -> np.ndarray:
        """
        Validate that input is a 1D numeric array of probabilities in [0.0, 1.0].

        Args:
            probs: 1D array or list of probabilities

        Returns:
            Validated 1D numpy array of float32
        """
        if not isinstance(probs, np.ndarray):
            try:
                arr = np.asarray(probs, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise InvalidProbabilityError(f"Cannot convert input to numeric array: {e}") from e
        else:
            arr = probs.astype(np.float32)

        if arr.ndim > 1:
            arr = arr.flatten()

        if len(arr) == 0:
            raise InvalidProbabilityError("Probability array is empty.")

        if np.isnan(arr).any():
            raise InvalidProbabilityError("Input probabilities contain NaN values.")

        if np.isinf(arr).any():
            raise InvalidProbabilityError("Input probabilities contain infinite values.")

        if (arr < 0.0).any() or (arr > 1.0).any():
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            raise InvalidProbabilityError(
                f"Probabilities must be within [0.0, 1.0], observed min={min_val}, max={max_val}. "
                "Ensure raw margin scores are not passed directly to the probability calibrator."
            )

        return arr

    def validate_targets(self, y: Any) -> np.ndarray:
        """Validate binary ground truth labels."""
        if not isinstance(y, np.ndarray):
            arr = np.asarray(y, dtype=np.int32)
        else:
            arr = y.astype(np.int32)

        if arr.ndim > 1:
            arr = arr.flatten()

        unique = set(np.unique(arr))
        if not unique.issubset({0, 1}):
            raise ValueError(f"Ground truth labels must be binary {0, 1}, got {unique}")

        return arr

    def fit(
        self,
        uncalibrated_probs: Union[np.ndarray, List[float]],
        y_true: Union[np.ndarray, List[int]],
    ) -> CalibrationResult:
        """
        Fit calibrator on uncalibrated model probabilities and binary labels.

        Args:
            uncalibrated_probs: Array of probabilities in [0.0, 1.0] from validation/calibration split
            y_true: Binary ground truth labels (0 or 1)

        Returns:
            CalibrationResult dataclass
        """
        probs = self.validate_probabilities(uncalibrated_probs)
        y = self.validate_targets(y_true)

        if len(probs) != len(y):
            raise ValueError(f"Sample count mismatch: {len(probs)} probabilities vs {len(y)} labels.")

        if len(probs) < 20:
            raise InsufficientDataError(
                f"Insufficient calibration data: {len(probs)} samples provided. Minimum required: 20."
            )

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise InsufficientDataError(
                f"Calibration dataset contains only one class: {unique_labels}. Both classes {0, 1} are required."
            )

        logger.info(
            f"Fitting {self.method} probability calibrator on {len(probs)} samples "
            f"(Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)})..."
        )

        # Initialize and fit calibrator
        if self.method == "sigmoid":
            self.calibrator = _SigmoidCalibration()
            self.calibrator.fit(probs, y)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(probs, y)

        self.is_fitted = True

        # Compute calibration metrics
        calibrated_probs = self.predict_proba(probs)

        brier_uncal = float(brier_score_loss(y, probs))
        brier_cal = float(brier_score_loss(y, calibrated_probs))
        loss_uncal = float(log_loss(y, probs))
        loss_cal = float(log_loss(y, calibrated_probs))
        ece_uncal, mce_uncal = compute_ece(y, probs)
        ece_cal, mce_cal = compute_ece(y, calibrated_probs)

        metrics = CalibrationMetrics(
            brier_score_uncalibrated=round(brier_uncal, 6),
            brier_score_calibrated=round(brier_cal, 6),
            log_loss_uncalibrated=round(loss_uncal, 6),
            log_loss_calibrated=round(loss_cal, 6),
            ece_uncalibrated=round(ece_uncal, 6),
            ece_calibrated=round(ece_cal, 6),
            mce_uncalibrated=round(mce_uncal, 6),
            mce_calibrated=round(mce_cal, 6),
            brier_improvement=round(brier_uncal - brier_cal, 6),
            log_loss_improvement=round(loss_uncal - loss_cal, 6),
            ece_improvement=round(ece_uncal - ece_cal, 6),
        )

        result = CalibrationResult(
            method=self.method,
            num_samples=len(probs),
            metrics=metrics,
            calibrator_file=self.calibrator_filename,
            metadata_file=self.metadata_filename,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_synthetic=True,
            disclaimer=CALIBRATION_SYNTHETIC_DISCLAIMER,
        )

        self.metadata = result.to_dict()
        logger.info(
            f"Calibration complete. Brier: {brier_uncal:.4f} -> {brier_cal:.4f}, "
            f"LogLoss: {loss_uncal:.4f} -> {loss_cal:.4f}, ECE: {ece_uncal:.4f} -> {ece_cal:.4f}"
        )

        return result

    def predict_proba(
        self,
        uncalibrated_probs: Union[float, np.ndarray, List[float]],
    ) -> np.ndarray:
        """
        Map uncalibrated probabilities to calibrated probabilities.

        Args:
            uncalibrated_probs: Array, list, or scalar of uncalibrated probabilities in [0.0, 1.0]

        Returns:
            1D array of calibrated probabilities in [0.0, 1.0]
        """
        if not self.is_fitted or self.calibrator is None:
            raise CalibratorNotFittedError("Calibrator is not fitted or loaded. Call fit() or load() first.")

        probs = self.validate_probabilities(uncalibrated_probs)

        # Transform probabilities
        if self.method == "sigmoid":
            cal_probs = self.calibrator.predict(probs)
        elif self.method == "isotonic":
            cal_probs = self.calibrator.predict(probs)
        else:
            raise CalibrationError(f"Unknown calibration method: {self.method}")

        # Ensure bounds [0.0, 1.0]
        cal_probs = np.clip(cal_probs, 0.0, 1.0).astype(np.float32)
        return cal_probs

    def calibrate_single(self, raw_prob: float) -> Tuple[float, float]:
        """
        Calibrate a single probability value.

        Args:
            raw_prob: Uncalibrated probability float in [0.0, 1.0]

        Returns:
            Tuple of (raw_probability, calibrated_probability)
        """
        cal = float(self.predict_proba([raw_prob])[0])
        return float(raw_prob), cal

    def save(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Save fitted calibrator object and calibration metadata.

        Args:
            output_dir: Output directory (defaults to models/)

        Returns:
            Tuple of (calibrator_path, metadata_path)
        """
        if not self.is_fitted or self.calibrator is None:
            raise CalibratorNotFittedError("Cannot save unfitted calibrator.")

        target_dir = Path(output_dir) if output_dir else self.models_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        cal_path = target_dir / self.calibrator_filename
        meta_path = target_dir / self.metadata_filename

        joblib.dump(self.calibrator, cal_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved calibrator to {cal_path} and metadata to {meta_path}")
        return cal_path, meta_path

    def load(
        self,
        calibrator_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load serialized calibrator object and metadata from disk.

        Args:
            calibrator_path: Path to calibrator.joblib
            metadata_path: Path to calibration_metadata.json
        """
        c_path = Path(calibrator_path) if calibrator_path else (self.models_dir / self.calibrator_filename)
        m_path = Path(metadata_path) if metadata_path else (self.models_dir / self.metadata_filename)

        if not c_path.exists():
            raise FileNotFoundError(f"Calibrator artifact not found at: {c_path}")

        try:
            self.calibrator = joblib.load(c_path)
            self.is_fitted = True
            logger.info(f"Loaded probability calibrator from {c_path}")
        except Exception as e:
            raise CalibrationError(f"Failed to load calibrator from {c_path}: {e}") from e

        if m_path.exists():
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.method = self.metadata.get("method", self.method)
            except Exception as e:
                logger.warning(f"Could not parse calibration metadata from {m_path}: {e}")
        else:
            self.metadata = {"method": self.method, "calibrator_file": c_path.name}


def fit_calibrator_from_splits(
    data_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    method: str = "sigmoid",
) -> Tuple[ProbabilityCalibrator, CalibrationResult]:
    """
    Convenience function to load calibration split (X_cal, y_cal), run inference
    with MalwareClassifier, fit the calibrator, and save artifacts.

    Args:
        data_dir: Directory containing X_cal.npy and y_cal.npy
        models_dir: Directory to save calibrator artifacts
        method: Calibration method ('sigmoid' or 'isotonic')

    Returns:
        Tuple of (fitted ProbabilityCalibrator, CalibrationResult)
    """
    d_dir = Path(data_dir) if data_dir else DATA_CONFIG.PROCESSED_DATA_DIR
    x_cal_path = d_dir / "X_cal.npy"
    y_cal_path = d_dir / "y_cal.npy"

    if not x_cal_path.exists() or not y_cal_path.exists():
        raise FileNotFoundError(f"Calibration split files missing in {d_dir} (expected X_cal.npy and y_cal.npy)")

    from threatlens.model.classifier import MalwareClassifier

    clf = MalwareClassifier(model_path=(Path(models_dir) if models_dir else DATA_CONFIG.MODELS_DIR) / MODEL_CONFIG.MODEL_FILENAME)
    X_cal = np.load(x_cal_path)
    y_cal = np.load(y_cal_path)

    # CRITICAL: Predict probabilities, NOT raw SHAP margin log-odds
    uncalibrated_probs = clf.predict_proba(X_cal)

    calibrator = ProbabilityCalibrator(method=method, models_dir=models_dir)
    result = calibrator.fit(uncalibrated_probs, y_cal)
    calibrator.save()

    return calibrator, result


__all__ = [
    "ProbabilityCalibrator",
    "CalibrationResult",
    "CalibrationMetrics",
    "CalibrationError",
    "CalibratorNotFittedError",
    "InsufficientDataError",
    "InvalidProbabilityError",
    "compute_ece",
    "fit_calibrator_from_splits",
    "CALIBRATION_SYNTHETIC_DISCLAIMER",
]
