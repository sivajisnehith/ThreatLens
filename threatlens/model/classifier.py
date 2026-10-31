"""
ThreatLens Malware Classifier Inference Interface

Provides high-performance, validated binary inference on the 2,351-dimensional
EMBER 2018 feature schema. Enforces strict dimensionality, ordering, and schema
isolation against custom PE feature inputs.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np

from threatlens.config import (
    CALIBRATION_CONFIG,
    DATA_CONFIG,
    MODEL_CONFIG,
    EmberFeatureGroups,
)
from threatlens.core.adapter import IncompatibleFeatureError
from threatlens.utils.logger import get_logger

logger = get_logger("classifier")


class ModelNotLoadedError(Exception):
    """Raised when inference is attempted on an uninitialized model."""
    pass


class FeatureDimensionError(Exception):
    """Raised when input feature vector does not conform to 2,351 dimensions."""
    pass


@dataclass
class PredictionResult:
    """Structured inference result for a single sample."""
    label: int
    label_name: str
    probability: float
    benign_probability: float
    threshold: float
    confidence: float
    feature_dim: int
    model_file: str
    calibrated_probability: Optional[float] = None
    is_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        return asdict(self)


@dataclass
class BatchPredictionResult:
    """Structured inference result for a batch of samples."""
    labels: np.ndarray
    probabilities: np.ndarray
    count: int
    threshold: float
    feature_dim: int
    calibrated_probabilities: Optional[np.ndarray] = None
    is_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch result to dictionary."""
        res: Dict[str, Any] = {
            "labels": self.labels.tolist(),
            "probabilities": self.probabilities.tolist(),
            "count": self.count,
            "threshold": self.threshold,
            "feature_dim": self.feature_dim,
            "is_calibrated": self.is_calibrated,
        }
        if self.calibrated_probabilities is not None:
            res["calibrated_probabilities"] = self.calibrated_probabilities.tolist()
        return res


class MalwareClassifier:
    """
    Primary binary classifier interface for ThreatLens static malware detection.
    Operates on the official 2,351-dimensional EMBER feature schema.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        calibrator_path: Optional[Union[str, Path]] = None,
        calibrator: Optional[Any] = None,
    ):
        """
        Initialize classifier by loading serialized model and metadata.

        Args:
            model_path: Path to serialized LightGBM model text file
            metadata_path: Path to model_metadata.json
            calibrator_path: Path to calibrator.joblib
            calibrator: Optional pre-initialized or fitted ProbabilityCalibrator instance
        """
        self.expected_dim: int = EmberFeatureGroups.TOTAL_FEATURES  # 2351
        self.model_path = Path(model_path) if model_path else (DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME)
        self.metadata_path = Path(metadata_path) if metadata_path else (DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.METADATA_FILENAME)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else (DATA_CONFIG.MODELS_DIR / CALIBRATION_CONFIG.CALIBRATOR_FILENAME)

        self.booster: Optional[lgb.Booster] = None
        self.metadata: Dict[str, Any] = {}
        self.calibrator: Optional[Any] = calibrator
        self.is_loaded: bool = False

        self.load_model()
        if self.calibrator is None and self.calibrator_path.exists():
            self.load_calibrator()

    def load_calibrator(self, calibrator_path: Optional[Union[str, Path]] = None) -> None:
        """Load probability calibrator from disk."""
        target_path = Path(calibrator_path) if calibrator_path else self.calibrator_path
        if target_path.exists():
            try:
                from threatlens.model.calibration import ProbabilityCalibrator
                cal = ProbabilityCalibrator(models_dir=target_path.parent)
                cal.load(calibrator_path=target_path)
                self.calibrator = cal
                logger.info(f"Loaded probability calibrator from {target_path}")
            except Exception as e:
                logger.warning(f"Could not load calibrator from {target_path}: {e}")
                self.calibrator = None

    def load_model(self) -> None:
        """Load LightGBM booster model and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model file not found at: {self.model_path}. "
                "Ensure Stage 3 model training has been executed."
            )

        try:
            self.booster = lgb.Booster(model_file=str(self.model_path))
            self.is_loaded = True
            logger.info(f"Loaded LightGBM model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM model: {e}")
            raise RuntimeError(f"Could not load LightGBM model from {self.model_path}: {e}") from e

        # Load optional metadata
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse model metadata from {self.metadata_path}: {e}")
        else:
            self.metadata = {
                "model_file": self.model_path.name,
                "feature_dim": self.expected_dim,
            }

    def validate_features(self, features: Any) -> np.ndarray:
        """
        Validate input dimensionality, reject custom PE schema dictionaries,
        and normalize into a 2D float32 numpy array of shape (N, 2351).

        Args:
            features: Input features (1D array, 2D array, or list of numbers)

        Returns:
            Validated numpy array of shape (N, 2351) with dtype float32
        """
        # Reject custom PE feature dictionaries
        if isinstance(features, dict):
            if "schema_version" in features or "dos_header" in features or "sections" in features:
                raise IncompatibleFeatureError(
                    "Incompatible feature schema: MalwareClassifier operates exclusively on the "
                    "2,351-dimensional EMBER 2018 feature schema. Received ThreatLens custom PE "
                    "feature dictionary. Cross-pipeline silent mapping is strictly prohibited."
                )
            raise IncompatibleFeatureError(
                f"MalwareClassifier requires numerical array of 2,351 features, got dict"
            )

        # Convert to numpy array
        if not isinstance(features, np.ndarray):
            try:
                arr = np.asarray(features, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not convert input to numeric array: {e}") from e
        else:
            arr = features.astype(np.float32)

        # Dimension checks
        if arr.ndim == 1:
            if len(arr) != self.expected_dim:
                raise FeatureDimensionError(
                    f"Feature dimension mismatch: expected {self.expected_dim} features, got {len(arr)}"
                )
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] != self.expected_dim:
                raise FeatureDimensionError(
                    f"Feature dimension mismatch: expected {self.expected_dim} features, got {arr.shape[1]}"
                )
        else:
            raise FeatureDimensionError(
                f"Features must be 1D or 2D array, got {arr.ndim}D array with shape {arr.shape}"
            )

        if np.isnan(arr).any():
            logger.warning("Input features contain NaN values. LightGBM will handle them via default branch.")
        if np.isinf(arr).any():
            raise ValueError("Input features contain infinite values, which are invalid for inference.")

        return arr

    def predict_proba(self, features: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """
        Return the raw model prediction probability of being malicious (class 1).

        Args:
            features: Input array of shape (2351,) or (N, 2351)

        Returns:
            1D array of shape (N,) containing probabilities in [0.0, 1.0]
        """
        if not self.is_loaded or self.booster is None:
            raise ModelNotLoadedError("Model is not loaded. Call load_model() first.")

        X = self.validate_features(features)
        probs = self.booster.predict(X)
        return probs.astype(np.float32)

    def predict(
        self,
        features: Union[np.ndarray, List[Any]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return discrete binary predictions (0 for benign, 1 for malicious).

        Args:
            features: Input array of shape (2351,) or (N, 2351)
            threshold: Decision threshold for malicious classification

        Returns:
            1D integer array of shape (N,) containing 0 or 1
        """
        probs = self.predict_proba(features)
        return (probs >= threshold).astype(int)

    def predict_calibrated_proba(
        self,
        features: Union[np.ndarray, List[Any]],
    ) -> np.ndarray:
        """
        Return calibrated probabilities of being malicious (class 1).
        Falls back to uncalibrated probabilities if calibrator is not loaded.

        Args:
            features: Input array of shape (2351,) or (N, 2351)

        Returns:
            1D array of shape (N,) containing calibrated probabilities in [0.0, 1.0]
        """
        raw_probs = self.predict_proba(features)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

    def predict_single(
        self,
        features: Union[np.ndarray, List[float]],
        threshold: float = 0.5,
    ) -> PredictionResult:
        """
        Perform inference on a single sample and return a structured PredictionResult.

        Args:
            features: Single feature vector of shape (2351,)
            threshold: Decision threshold

        Returns:
            PredictionResult object
        """
        probs = self.predict_proba(features)
        mal_prob = float(probs[0])
        label = 1 if mal_prob >= threshold else 0
        label_name = "MALICIOUS" if label == 1 else "BENIGN"
        confidence = mal_prob if label == 1 else (1.0 - mal_prob)

        cal_prob: Optional[float] = None
        is_calibrated = False
        if self.calibrator is not None:
            try:
                cal_prob = float(self.calibrator.predict_proba([mal_prob])[0])
                is_calibrated = True
            except Exception as e:
                logger.warning(f"Failed to calibrate probability: {e}")

        return PredictionResult(
            label=label,
            label_name=label_name,
            probability=mal_prob,
            benign_probability=1.0 - mal_prob,
            threshold=threshold,
            confidence=confidence,
            feature_dim=self.expected_dim,
            model_file=self.model_path.name,
            calibrated_probability=cal_prob,
            is_calibrated=is_calibrated,
        )

    def predict_batch(
        self,
        features: Union[np.ndarray, List[List[float]]],
        threshold: float = 0.5,
    ) -> BatchPredictionResult:
        """
        Perform batch inference on multiple samples.

        Args:
            features: Feature matrix of shape (N, 2351)
            threshold: Decision threshold

        Returns:
            BatchPredictionResult object
        """
        probs = self.predict_proba(features)
        labels = (probs >= threshold).astype(int)

        cal_probs: Optional[np.ndarray] = None
        is_calibrated = False
        if self.calibrator is not None:
            try:
                cal_probs = self.calibrator.predict_proba(probs)
                is_calibrated = True
            except Exception as e:
                logger.warning(f"Failed to calibrate batch probabilities: {e}")

        return BatchPredictionResult(
            labels=labels,
            probabilities=probs,
            count=len(probs),
            threshold=threshold,
            feature_dim=self.expected_dim,
            calibrated_probabilities=cal_probs,
            is_calibrated=is_calibrated,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return dict(self.metadata)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of 2,351 model feature names."""
        if not self.is_loaded or self.booster is None:
            raise ModelNotLoadedError("Model is not loaded.")
        return self.booster.feature_name()


__all__ = [
    "MalwareClassifier",
    "PredictionResult",
    "BatchPredictionResult",
    "ModelNotLoadedError",
    "FeatureDimensionError",
]
