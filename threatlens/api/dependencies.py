"""
ThreatLens API Dependency Injection and Service State Container

Initializes the primary classifier, probability calibrator, SHAP explainer, and evaluation
metrics once during startup. Injects them cleanly into route handlers without recalculation.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from threatlens.config import CALIBRATION_CONFIG, DATA_CONFIG, MODEL_CONFIG
from threatlens.explainability.explainer import ShapExplainer
from threatlens.model.calibration import ProbabilityCalibrator
from threatlens.model.classifier import MalwareClassifier
from threatlens.utils.logger import get_logger

logger = get_logger("api_dependencies")


class ModelContainer:
    """
    Singleton application state container for loaded model artifacts.
    Prevents repeated disk I/O and tree initialization across HTTP requests.
    """

    def __init__(self):
        self.classifier: Optional[MalwareClassifier] = None
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.explainer: Optional[ShapExplainer] = None
        self.evaluation_report: Optional[Dict[str, Any]] = None
        self.is_ready: bool = False

    def load_all(
        self,
        model_path: Optional[Path] = None,
        calibrator_path: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
    ) -> None:
        """Load all model artifacts into memory."""
        m_path = model_path or (DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME)
        c_path = calibrator_path or (DATA_CONFIG.MODELS_DIR / CALIBRATION_CONFIG.CALIBRATOR_FILENAME)
        r_dir = reports_dir or DATA_CONFIG.REPORTS_DIR

        logger.info("Initializing ThreatLens ModelContainer services...")

        # 1. Load Calibrator if present
        if c_path.exists():
            try:
                self.calibrator = ProbabilityCalibrator(models_dir=c_path.parent)
                self.calibrator.load(calibrator_path=c_path)
                logger.info(f"Loaded probability calibrator from {c_path}")
            except Exception as e:
                logger.warning(f"Could not load calibrator from {c_path}: {e}")
                self.calibrator = None
        else:
            logger.info("No calibrator artifact found; running in uncalibrated mode.")
            self.calibrator = None

        # 2. Load Primary Classifier
        if m_path.exists():
            self.classifier = MalwareClassifier(
                model_path=m_path,
                calibrator=self.calibrator,
            )
            logger.info(f"Loaded MalwareClassifier from {m_path}")
        else:
            logger.error(f"Trained model not found at {m_path}")
            raise FileNotFoundError(f"Model file not found at {m_path}")

        # 3. Load SHAP Explainer
        try:
            self.explainer = ShapExplainer(
                model=self.classifier.booster,
                calibrator=self.calibrator,
            )
            logger.info("Loaded ShapExplainer instance successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ShapExplainer: {e}")
            raise RuntimeError(f"Failed to initialize ShapExplainer: {e}") from e

        # 4. Load Evaluation Report
        eval_path = r_dir / "classification_report.json"
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    self.evaluation_report = json.load(f)
                logger.info(f"Loaded evaluation report from {eval_path}")
            except Exception as e:
                logger.warning(f"Could not load evaluation report: {e}")
                self.evaluation_report = None
        else:
            self.evaluation_report = None

        self.is_ready = True
        logger.info("ModelContainer ready to serve inference requests.")


# Global singleton instance
container = ModelContainer()


def get_model_container() -> ModelContainer:
    """Dependency provider for the full ModelContainer."""
    return container


def get_classifier() -> MalwareClassifier:
    """Dependency provider for MalwareClassifier."""
    if not container.is_ready or container.classifier is None:
        raise RuntimeError("Classifier service is unavailable.")
    return container.classifier


def get_calibrator() -> Optional[ProbabilityCalibrator]:
    """Dependency provider for ProbabilityCalibrator."""
    return container.calibrator


def get_explainer() -> ShapExplainer:
    """Dependency provider for ShapExplainer."""
    if not container.is_ready or container.explainer is None:
        raise RuntimeError("Explainer service is unavailable.")
    return container.explainer
