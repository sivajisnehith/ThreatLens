"""
ThreatLens SHAP Explainability Engine

Integrates SHAP TreeExplainer with the trained LightGBM malware classifier.
Provides verified local and batch explanations, enforces strict schema isolation,
validates output dimensionality, checks exact additivity, and marks synthetic-model
limitations clearly.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import shap

from threatlens.config import DATA_CONFIG, MODEL_CONFIG, EmberFeatureGroups
from threatlens.core.adapter import IncompatibleFeatureError
from threatlens.explainability.feature_interpreter import (
    DIRECTION_DECREASES_MALWARE,
    DIRECTION_INCREASES_MALWARE,
    DIRECTION_NEUTRAL,
    FeatureContribution,
    FeatureInterpreter,
)
from threatlens.model.classifier import (
    FeatureDimensionError,
    MalwareClassifier,
    ModelNotLoadedError,
)
from threatlens.model.trainer import generate_ember_feature_names
from threatlens.utils.logger import get_logger

logger = get_logger("shap_explainer")

SYNTHETIC_MODEL_DISCLAIMER: str = (
    "DISCLAIMER: Explanations are produced from a model trained on synthetic data "
    "for validation purposes. Identified feature attributions reflect synthetic distributions "
    "and must NOT be interpreted as real-world malware indicators or cybersecurity discoveries."
)


class ExplainerNotInitializedError(Exception):
    """Raised when an explanation is attempted before initialization."""
    pass


class ExplainerModelError(Exception):
    """Raised when the model cannot be loaded or is incompatible with TreeExplainer."""
    pass


class InvalidFeatureError(Exception):
    """Raised when input feature vector is invalid for explanation."""
    pass


@dataclass
class SingleExplanation:
    """
    Structured local explanation for a single prediction.
    """
    prediction: int
    prediction_label: str
    probability: float
    benign_probability: float
    raw_score: float
    expected_value: float
    threshold: float
    top_positive_contributors: List[FeatureContribution]
    top_negative_contributors: List[FeatureContribution]
    top_contributors: List[FeatureContribution]
    feature_values: List[float]
    shap_values: List[float]
    feature_names: List[str]
    additivity_error: float
    calibrated_probability: Optional[float] = None
    is_calibrated: bool = False
    is_synthetic_model: bool = True
    disclaimer: str = SYNTHETIC_MODEL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to JSON-serializable dictionary."""
        return {
            "prediction": int(self.prediction),
            "prediction_label": str(self.prediction_label),
            "probability": float(self.probability),
            "benign_probability": float(self.benign_probability),
            "raw_score": float(self.raw_score),
            "expected_value": float(self.expected_value),
            "threshold": float(self.threshold),
            "additivity_error": float(self.additivity_error),
            "calibrated_probability": float(self.calibrated_probability) if self.calibrated_probability is not None else None,
            "is_calibrated": bool(self.is_calibrated),
            "is_synthetic_model": bool(self.is_synthetic_model),
            "disclaimer": str(self.disclaimer),
            "top_positive_contributors": [c.to_dict() for c in self.top_positive_contributors],
            "top_negative_contributors": [c.to_dict() for c in self.top_negative_contributors],
            "top_contributors": [c.to_dict() for c in self.top_contributors],
            "feature_values": [float(v) for v in self.feature_values],
            "shap_values": [float(v) for v in self.shap_values],
            "feature_names": list(self.feature_names),
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize explanation to JSON string, optionally writing to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath is not None:
            out_path = Path(filepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Saved explanation JSON to {out_path}")
        return json_str


@dataclass
class BatchExplanation:
    """
    Structured batch explanation container.
    """
    count: int
    predictions: List[int]
    probabilities: List[float]
    raw_scores: List[float]
    expected_value: float
    feature_names: List[str]
    shap_matrix: np.ndarray
    feature_matrix: np.ndarray
    mean_abs_shap: List[float]
    threshold: float
    is_synthetic_model: bool = True
    disclaimer: str = SYNTHETIC_MODEL_DISCLAIMER

    def get_explanation(
        self,
        index: int,
        interpreter: FeatureInterpreter,
        top_k: int = 20,
    ) -> SingleExplanation:
        """
        Extract SingleExplanation for a specific sample in the batch.

        Args:
            index: Index in [0, count-1]
            interpreter: FeatureInterpreter instance
            top_k: Top K contributors to retain

        Returns:
            SingleExplanation
        """
        if not (0 <= index < self.count):
            raise IndexError(f"Sample index {index} out of range [0, {self.count - 1}]")

        x_row = self.feature_matrix[index]
        shap_row = self.shap_matrix[index]
        prob = self.probabilities[index]
        raw = self.raw_scores[index]
        pred = self.predictions[index]
        pred_label = "MALICIOUS" if pred == 1 else "BENIGN"

        additivity_error = float(abs((self.expected_value + np.sum(shap_row)) - raw))

        # Build feature contributions
        contributions: List[FeatureContribution] = []
        for i in range(len(shap_row)):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            contrib = interpreter.interpret_contribution(
                feature_index=i,
                feature_value=float(x_row[i]),
                shap_value=float(shap_row[i]),
                feature_name=name,
            )
            contributions.append(contrib)

        # Sort by absolute SHAP descending
        sorted_by_abs = sorted(contributions, key=lambda c: c.abs_shap_value, reverse=True)
        for rank, c in enumerate(sorted_by_abs, start=1):
            c.rank = rank

        top_abs = sorted_by_abs[:top_k]

        pos_contributors = [c for c in sorted_by_abs if c.shap_value > 1e-7][:top_k]
        neg_contributors = [c for c in sorted_by_abs if c.shap_value < -1e-7][:top_k]

        return SingleExplanation(
            prediction=pred,
            prediction_label=pred_label,
            probability=prob,
            benign_probability=1.0 - prob,
            raw_score=raw,
            expected_value=self.expected_value,
            threshold=self.threshold,
            top_positive_contributors=pos_contributors,
            top_negative_contributors=neg_contributors,
            top_contributors=top_abs,
            feature_values=[float(v) for v in x_row],
            shap_values=[float(v) for v in shap_row],
            feature_names=self.feature_names,
            additivity_error=additivity_error,
            is_synthetic_model=self.is_synthetic_model,
            disclaimer=self.disclaimer,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch summary to JSON-serializable dictionary."""
        return {
            "count": int(self.count),
            "threshold": float(self.threshold),
            "expected_value": float(self.expected_value),
            "predictions": [int(p) for p in self.predictions],
            "probabilities": [float(p) for p in self.probabilities],
            "raw_scores": [float(s) for s in self.raw_scores],
            "mean_abs_shap": [float(m) for m in self.mean_abs_shap],
            "feature_names": list(self.feature_names),
            "is_synthetic_model": bool(self.is_synthetic_model),
            "disclaimer": str(self.disclaimer),
        }


class ShapExplainer:
    """
    Primary SHAP explainability engine for ThreatLens.
    Wraps LightGBM models with SHAP TreeExplainer, enforcing exact feature mapping,
    strict schema validation, additivity verification, and robust multi-version compatibility.
    """

    def __init__(
        self,
        model: Optional[Union[lgb.Booster, MalwareClassifier, str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        feature_names: Optional[List[str]] = None,
        calibrator: Optional[Any] = None,
        calibrator_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the SHAP explainability engine.

        Args:
            model: LightGBM Booster instance, MalwareClassifier instance, or path to model file.
                   If None, defaults to DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME.
            metadata_path: Optional path to model_metadata.json
            feature_names: Optional ordered list of 2,351 feature names.
            calibrator: Optional ProbabilityCalibrator instance.
            calibrator_path: Optional path to calibrator.joblib artifact.
        """
        self.expected_dim: int = EmberFeatureGroups.TOTAL_FEATURES  # 2351
        self.booster: Optional[lgb.Booster] = None
        self.metadata: Dict[str, Any] = {}
        self.explainer: Optional[shap.TreeExplainer] = None
        self.expected_value: float = 0.0
        self.feature_names: List[str] = []
        self.interpreter: Optional[FeatureInterpreter] = None
        self.is_synthetic_model: bool = True
        self.calibrator: Optional[Any] = calibrator

        self._load_model_and_metadata(model, metadata_path)
        self._init_feature_names(feature_names)
        self._init_tree_explainer()
        self.interpreter = FeatureInterpreter(feature_names=self.feature_names)

        # Attempt to load calibrator if not explicitly provided
        if self.calibrator is None:
            c_path = Path(calibrator_path) if calibrator_path else (DATA_CONFIG.MODELS_DIR / "calibrator.joblib")
            if c_path.exists():
                try:
                    from threatlens.model.calibration import ProbabilityCalibrator
                    cal = ProbabilityCalibrator(models_dir=c_path.parent)
                    cal.load(calibrator_path=c_path)
                    self.calibrator = cal
                except Exception as e:
                    logger.debug(f"Could not automatically load calibrator: {e}")

    def _load_model_and_metadata(
        self,
        model: Optional[Union[lgb.Booster, MalwareClassifier, str, Path]],
        metadata_path: Optional[Union[str, Path]],
    ) -> None:
        """Load Booster model and associated metadata."""
        if model is None:
            default_path = DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME
            if not default_path.exists():
                raise FileNotFoundError(
                    f"LightGBM model not found at default location: {default_path}. "
                    "Train the model or specify a valid model path."
                )
            try:
                self.booster = lgb.Booster(model_file=str(default_path))
            except Exception as e:
                raise ExplainerModelError(f"Failed to load Booster from {default_path}: {e}") from e
            resolved_meta = metadata_path or (DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.METADATA_FILENAME)

        elif isinstance(model, MalwareClassifier):
            if not model.is_loaded or model.booster is None:
                raise ExplainerModelError("Provided MalwareClassifier instance is not loaded.")
            self.booster = model.booster
            self.metadata = model.get_metadata()
            return

        elif isinstance(model, lgb.Booster):
            self.booster = model
            resolved_meta = metadata_path

        elif isinstance(model, (str, Path)):
            m_path = Path(model)
            if not m_path.exists():
                raise FileNotFoundError(f"Model file not found at: {m_path}")
            try:
                self.booster = lgb.Booster(model_file=str(m_path))
            except Exception as e:
                raise ExplainerModelError(f"Failed to load Booster from {m_path}: {e}") from e
            resolved_meta = metadata_path or (m_path.parent / MODEL_CONFIG.METADATA_FILENAME)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        # Load metadata if available
        if resolved_meta and Path(resolved_meta).exists():
            try:
                with open(resolved_meta, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata from {resolved_meta}: {e}")
        else:
            self.metadata = {
                "feature_dim": self.expected_dim,
                "model_type": "LightGBM",
            }

        # Check provenance notes
        prov = self.metadata.get("provenance_notes", "").lower()
        if "synthetic" in prov or "synthetic" in self.metadata.get("dataset_type", "").lower():
            self.is_synthetic_model = True

    def _init_feature_names(self, custom_names: Optional[List[str]]) -> None:
        """Resolve and validate canonical feature names."""
        if custom_names is not None:
            if len(custom_names) != self.expected_dim:
                raise ValueError(
                    f"Provided feature names list length ({len(custom_names)}) does not "
                    f"match required EMBER dimension {self.expected_dim}"
                )
            self.feature_names = list(custom_names)
            return

        # Attempt to read feature names directly from Booster
        booster_names = self.booster.feature_name() if self.booster else []
        if (
            booster_names
            and len(booster_names) == self.expected_dim
            and not booster_names[0].startswith("Column_")
        ):
            self.feature_names = list(booster_names)
        else:
            # Generate canonical EMBER feature names
            self.feature_names = generate_ember_feature_names()

    def _init_tree_explainer(self) -> None:
        """Initialize SHAP TreeExplainer and normalize expected value."""
        if self.booster is None:
            raise ExplainerModelError("Booster is not initialized.")

        try:
            # TreeExplainer with model_output='raw' ensures exact log-odds additivity
            self.explainer = shap.TreeExplainer(self.booster, model_output="raw")
        except Exception as e:
            raise ExplainerModelError(f"Failed to initialize shap.TreeExplainer: {e}") from e

        # Normalize expected_value
        ev = self.explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            ev_arr = np.asarray(ev).flatten()
            if len(ev_arr) == 1:
                self.expected_value = float(ev_arr[0])
            elif len(ev_arr) == 2:
                # Class 1 (malicious) baseline
                self.expected_value = float(ev_arr[1])
            else:
                self.expected_value = float(ev_arr[-1])
        else:
            self.expected_value = float(ev)

        logger.info(
            f"Initialized SHAP TreeExplainer. Feature dim: {self.expected_dim}, "
            f"Baseline expected value: {self.expected_value:.6f}"
        )

    def validate_features(self, features: Any) -> np.ndarray:
        """
        Strictly validate input feature dimensions, type, and reject custom PE schemas.

        Args:
            features: 1D array, 2D array, or list of numbers

        Returns:
            Validated 2D numpy array of shape (N, 2351) with dtype float32
        """
        # Strict isolation against custom PE feature schemas
        if isinstance(features, dict):
            if any(k in features for k in ("schema_version", "dos_header", "sections", "file_size")):
                raise IncompatibleFeatureError(
                    "Incompatible feature schema: ThreatLens ShapExplainer operates exclusively on "
                    "the 2,351-dimensional EMBER 2018 feature schema. Received custom PE feature "
                    "dictionary. Cross-pipeline mapping is strictly prohibited."
                )
            raise IncompatibleFeatureError("ShapExplainer requires numeric feature arrays, got dict")

        if not isinstance(features, np.ndarray):
            try:
                arr = np.asarray(features, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not convert input features to numeric array: {e}") from e
        else:
            arr = features.astype(np.float32)

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

        if np.isinf(arr).any():
            raise ValueError("Input features contain infinite values, which are invalid for inference.")

        if np.isnan(arr).any():
            logger.warning("Input features contain NaN values. Handled via default split branch.")

        return arr

    def _normalize_shap_output(self, raw_shap: Any, n_samples: int) -> np.ndarray:
        """
        Normalize SHAP output dimensions across different SHAP / LightGBM versions.
        Guarantees returned shape is (N, 2351) representing class 1 (malicious) attributions.

        Args:
            raw_shap: Output from explainer.shap_values()
            n_samples: Number of samples in batch

        Returns:
            2D numpy array of shape (N, 2351)
        """
        if isinstance(raw_shap, list):
            # Version returned list of ndarrays [class_0, class_1]
            if len(raw_shap) == 2:
                arr = np.asarray(raw_shap[1], dtype=np.float32)
            elif len(raw_shap) == 1:
                arr = np.asarray(raw_shap[0], dtype=np.float32)
            else:
                arr = np.asarray(raw_shap[-1], dtype=np.float32)
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                # Shape (N, 2351, 2)
                arr = raw_shap[:, :, 1].astype(np.float32)
            elif raw_shap.ndim == 2:
                # Shape (N, 2351)
                arr = raw_shap.astype(np.float32)
            elif raw_shap.ndim == 1:
                # Shape (2351,)
                arr = raw_shap.reshape(1, -1).astype(np.float32)
            else:
                raise ValueError(f"Unexpected SHAP array dimensions: {raw_shap.shape}")
        else:
            raise TypeError(f"Unexpected SHAP output type: {type(raw_shap)}")

        # Ensure correct shape
        if arr.shape != (n_samples, self.expected_dim):
            if arr.size == n_samples * self.expected_dim:
                arr = arr.reshape(n_samples, self.expected_dim)
            else:
                raise ValueError(
                    f"Normalized SHAP shape mismatch: expected ({n_samples}, {self.expected_dim}), "
                    f"got {arr.shape}"
                )

        return arr

    def compute_shap_values(self, features: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """
        Compute normalized SHAP attribution matrix of shape (N, 2351).

        Args:
            features: 1D or 2D array of features

        Returns:
            Array of shape (N, 2351)
        """
        if self.explainer is None:
            raise ExplainerNotInitializedError("TreeExplainer is not initialized.")

        X = self.validate_features(features)
        raw_sv = self.explainer.shap_values(X)
        normalized = self._normalize_shap_output(raw_sv, len(X))
        return normalized

    def explain_single(
        self,
        feature_vector: Union[np.ndarray, List[float]],
        top_k: int = 20,
        threshold: float = 0.5,
    ) -> SingleExplanation:
        """
        Generate detailed, structured local SHAP explanation for a single prediction.

        Args:
            feature_vector: 1D array of shape (2351,) or 2D array of shape (1, 2351)
            top_k: Number of top positive and negative contributors to include
            threshold: Decision threshold for malware classification

        Returns:
            SingleExplanation object
        """
        if self.booster is None or self.explainer is None or self.interpreter is None:
            raise ExplainerNotInitializedError("Explainer is not fully initialized.")

        X = self.validate_features(feature_vector)
        x_row = X[0]

        # Model predictions
        prob = float(self.booster.predict(X)[0])
        raw_score = float(self.booster.predict(X, raw_score=True)[0])
        pred = 1 if prob >= threshold else 0
        pred_label = "MALICIOUS" if pred == 1 else "BENIGN"

        # Compute SHAP values
        shap_row = self.compute_shap_values(X)[0]

        # Additivity verification
        shap_sum = float(np.sum(shap_row))
        additivity_error = float(abs((self.expected_value + shap_sum) - raw_score))
        if additivity_error > 1e-3:
            logger.warning(
                f"SHAP additivity difference observed: expected+sum={self.expected_value + shap_sum:.6f}, "
                f"raw_score={raw_score:.6f}, delta={additivity_error:.6f}"
            )

        # Interpret all contributions
        contributions: List[FeatureContribution] = []
        for i in range(self.expected_dim):
            name = self.feature_names[i]
            contrib = self.interpreter.interpret_contribution(
                feature_index=i,
                feature_value=float(x_row[i]),
                shap_value=float(shap_row[i]),
                feature_name=name,
            )
            contributions.append(contrib)

        # Sort by absolute contribution descending
        sorted_by_abs = sorted(contributions, key=lambda c: c.abs_shap_value, reverse=True)
        for rank, c in enumerate(sorted_by_abs, start=1):
            c.rank = rank

        top_abs = sorted_by_abs[:top_k]

        # Separate positive (increases malware risk) and negative (decreases malware risk)
        pos_contributors = [c for c in sorted_by_abs if c.shap_value > 1e-7][:top_k]
        neg_contributors = [c for c in sorted_by_abs if c.shap_value < -1e-7][:top_k]

        cal_prob: Optional[float] = None
        is_calibrated = False
        if self.calibrator is not None:
            try:
                cal_prob = float(self.calibrator.predict_proba([prob])[0])
                is_calibrated = True
            except Exception as e:
                logger.debug(f"Failed to calculate calibrated probability: {e}")

        return SingleExplanation(
            prediction=pred,
            prediction_label=pred_label,
            probability=prob,
            benign_probability=1.0 - prob,
            raw_score=raw_score,
            expected_value=self.expected_value,
            threshold=threshold,
            top_positive_contributors=pos_contributors,
            top_negative_contributors=neg_contributors,
            top_contributors=top_abs,
            feature_values=[float(v) for v in x_row],
            shap_values=[float(v) for v in shap_row],
            feature_names=self.feature_names,
            additivity_error=additivity_error,
            calibrated_probability=cal_prob,
            is_calibrated=is_calibrated,
            is_synthetic_model=self.is_synthetic_model,
            disclaimer=SYNTHETIC_MODEL_DISCLAIMER,
        )

    def explain_batch(
        self,
        feature_matrix: Union[np.ndarray, List[List[float]]],
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> BatchExplanation:
        """
        Generate explanations for a batch of samples efficiently.

        Args:
            feature_matrix: Matrix of shape (N, 2351)
            top_k: Number of top contributors per sample
            threshold: Decision threshold

        Returns:
            BatchExplanation container
        """
        if self.booster is None or self.explainer is None:
            raise ExplainerNotInitializedError("Explainer is not initialized.")

        X = self.validate_features(feature_matrix)
        count = len(X)

        probs = self.booster.predict(X).astype(float)
        raw_scores = self.booster.predict(X, raw_score=True).astype(float)
        preds = (probs >= threshold).astype(int)

        shap_matrix = self.compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0).astype(float).tolist()

        return BatchExplanation(
            count=count,
            predictions=preds.tolist(),
            probabilities=probs.tolist(),
            raw_scores=raw_scores.tolist(),
            expected_value=self.expected_value,
            feature_names=self.feature_names,
            shap_matrix=shap_matrix,
            feature_matrix=X,
            mean_abs_shap=mean_abs_shap,
            threshold=threshold,
            is_synthetic_model=self.is_synthetic_model,
            disclaimer=SYNTHETIC_MODEL_DISCLAIMER,
        )


__all__ = [
    "ShapExplainer",
    "SingleExplanation",
    "BatchExplanation",
    "ExplainerNotInitializedError",
    "ExplainerModelError",
    "InvalidFeatureError",
    "SYNTHETIC_MODEL_DISCLAIMER",
    "DIRECTION_INCREASES_MALWARE",
    "DIRECTION_DECREASES_MALWARE",
    "DIRECTION_NEUTRAL",
]
