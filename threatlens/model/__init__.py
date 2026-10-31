"""
ThreatLens Model Subpackage

LightGBM binary classification engine, evaluation suite, and inference interface
for the official EMBER 2018 2,351-dimensional feature schema.
"""

from threatlens.model.calibration import (
    CALIBRATION_SYNTHETIC_DISCLAIMER,
    CalibrationError,
    CalibrationMetrics,
    CalibrationResult,
    CalibratorNotFittedError,
    InsufficientDataError,
    InvalidProbabilityError,
    ProbabilityCalibrator,
    compute_ece,
    fit_calibrator_from_splits,
)
from threatlens.model.classifier import (
    BatchPredictionResult,
    FeatureDimensionError,
    MalwareClassifier,
    ModelNotLoadedError,
    PredictionResult,
)
from threatlens.model.evaluator import (
    EvaluationMetrics,
    ModelEvaluator,
)
from threatlens.model.threshold_analysis import (
    CYBERSECURITY_TRADEOFF_EXPLANATION,
    OperationalPosture,
    ThresholdAnalysisResult,
    ThresholdEvaluator,
    ThresholdPoint,
)
from threatlens.model.trainer import (
    ModelTrainer,
    TrainingResult,
    generate_ember_feature_names,
)

__all__ = [
    # Training
    "ModelTrainer",
    "TrainingResult",
    "generate_ember_feature_names",
    # Evaluation
    "ModelEvaluator",
    "EvaluationMetrics",
    # Inference
    "MalwareClassifier",
    "PredictionResult",
    "BatchPredictionResult",
    "ModelNotLoadedError",
    "FeatureDimensionError",
    # Calibration
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
    # Threshold Analysis
    "ThresholdEvaluator",
    "ThresholdAnalysisResult",
    "ThresholdPoint",
    "OperationalPosture",
    "CYBERSECURITY_TRADEOFF_EXPLANATION",
]
