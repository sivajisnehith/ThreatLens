"""
ThreatLens LightGBM Model Trainer

Handles training of the primary gradient boosted binary classifier on the
official EMBER 2018 2,351-dimensional feature representation. Enforces validation
early stopping, reproducible random seeds, and comprehensive metadata tracking.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np

from threatlens.config import (
    DATA_CONFIG,
    FEATURE_GROUPS,
    MODEL_CONFIG,
    EmberFeatureGroups,
    ModelConfig,
)
from threatlens.utils.logger import get_logger

logger = get_logger("model_trainer")


@dataclass
class TrainingResult:
    """Container for trained model artifacts and execution metrics."""
    booster: lgb.Booster
    best_iteration: int
    best_score: float
    train_duration_sec: float
    feature_names: List[str]
    config: ModelConfig
    train_samples: int
    val_samples: int
    evals_result: Dict[str, Any]
    timestamp: str


def generate_ember_feature_names() -> List[str]:
    """Generate the canonical 2,351 EMBER feature names matching the 8 groups."""
    names = []
    for group_name in FEATURE_GROUPS.get_all_group_names():
        start, end = FEATURE_GROUPS.get_group_indices(group_name)
        for i in range(start, end):
            names.append(f"ember_{group_name}_{i}")
    return names


class ModelTrainer:
    """
    Trainer for ThreatLens LightGBM binary classifier on EMBER 2018 features.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        models_dir: Optional[Path] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: ModelConfig instance (defaults to global MODEL_CONFIG)
            models_dir: Directory to save serialized models
        """
        self.config = config or MODEL_CONFIG
        self.models_dir = models_dir or DATA_CONFIG.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def validate_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "train",
    ) -> None:
        """Validate feature dimensions and binary labels."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(f"{split_name} data must be numpy arrays")

        if X.ndim != 2:
            raise ValueError(f"{split_name} features must be 2D array, got shape {X.shape}")

        if X.shape[1] != EmberFeatureGroups.TOTAL_FEATURES:
            raise ValueError(
                f"{split_name} feature dimension mismatch: expected {EmberFeatureGroups.TOTAL_FEATURES} "
                f"features, got {X.shape[1]}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"{split_name} sample count mismatch: {len(X)} features vs {len(y)} labels"
            )

        unique_labels = set(np.unique(y))
        valid_labels = {0, 1}
        if not unique_labels.issubset(valid_labels):
            raise ValueError(
                f"{split_name} contains invalid labels: {unique_labels}. Expected binary labels {valid_labels}"
            )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """
        Train LightGBM binary classifier with early stopping on validation set.

        Args:
            X_train: Training features array of shape (N_train, 2351)
            y_train: Training labels array of shape (N_train,)
            X_val: Validation features array of shape (N_val, 2351)
            y_val: Validation labels array of shape (N_val,)
            feature_names: Optional list of 2351 feature names

        Returns:
            TrainingResult containing booster and metadata
        """
        self.validate_dataset(X_train, y_train, "Training")
        self.validate_dataset(X_val, y_val, "Validation")

        if feature_names is None:
            feature_names = generate_ember_feature_names()
        elif len(feature_names) != EmberFeatureGroups.TOTAL_FEATURES:
            raise ValueError(
                f"Feature names list length ({len(feature_names)}) != {EmberFeatureGroups.TOTAL_FEATURES}"
            )

        logger.info(
            f"Initiating LightGBM training on {len(X_train)} samples with {X_train.shape[1]} features. "
            f"Validation set size: {len(X_val)}"
        )

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            reference=dtrain,
            feature_name=feature_names,
            free_raw_data=False,
        )

        # Build LightGBM parameter dictionary
        params: Dict[str, Any] = {
            "boosting_type": self.config.BOOSTING_TYPE,
            "objective": self.config.OBJECTIVE,
            "metric": list(self.config.METRIC),
            "learning_rate": self.config.LEARNING_RATE,
            "num_leaves": self.config.NUM_LEAVES,
            "max_depth": self.config.MAX_DEPTH,
            "min_data_in_leaf": self.config.MIN_DATA_IN_LEAF,
            "feature_fraction": self.config.FEATURE_FRACTION,
            "bagging_fraction": self.config.BAGGING_FRACTION,
            "bagging_freq": self.config.BAGGING_FREQ,
            "random_state": self.config.RANDOM_SEED,
            "verbosity": self.config.VERBOSITY,
            "is_unbalance": self.config.IS_UNBALANCE,
        }

        evals_result: Dict[str, Any] = {}
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config.EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.record_evaluation(evals_result),
        ]

        start_time = time.time()
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.config.NUM_BOOST_ROUND,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        duration = time.time() - start_time

        best_iter = booster.best_iteration
        best_score = float(evals_result.get("val", {}).get("binary_logloss", [0.0])[best_iter - 1]) if best_iter > 0 else 0.0

        logger.info(
            f"Training complete in {duration:.2f}s. Best iteration: {best_iter}, "
            f"Best validation binary_logloss: {best_score:.4f}"
        )

        return TrainingResult(
            booster=booster,
            best_iteration=best_iter,
            best_score=best_score,
            train_duration_sec=duration,
            feature_names=feature_names,
            config=self.config,
            train_samples=len(X_train),
            val_samples=len(X_val),
            evals_result=evals_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def save_model(
        self,
        result: TrainingResult,
        output_dir: Optional[Path] = None,
        provenance_notes: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Serialize model in text format and record complete model metadata.

        Args:
            result: TrainingResult instance
            output_dir: Directory to save model files
            provenance_notes: Optional notes regarding dataset provenance

        Returns:
            Tuple of (model_file_path, metadata_file_path)
        """
        target_dir = output_dir or self.models_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / self.config.MODEL_FILENAME
        metadata_path = target_dir / self.config.METADATA_FILENAME

        # Save Booster model in portable LightGBM text format
        result.booster.save_model(str(model_path), num_iteration=result.best_iteration)
        logger.info(f"Model saved to {model_path}")

        # Construct comprehensive metadata
        is_genuine = "genuine" in (provenance_notes or "").lower()
        metadata = {
            "threatlens_stage": 7 if is_genuine else 3,
            "model_type": "LightGBM",
            "model_format": "text",
            "model_file": self.config.MODEL_FILENAME,
            "created_at": result.timestamp,
            "feature_pipeline": "EMBER_2018",
            "feature_dim": EmberFeatureGroups.TOTAL_FEATURES,
            "feature_groups_count": len(FEATURE_GROUPS.get_all_group_names()),
            "best_iteration": result.best_iteration,
            "best_val_loss": result.best_score,
            "training_duration_seconds": round(result.train_duration_sec, 3),
            "samples": {
                "train_samples": result.train_samples,
                "val_samples": result.val_samples,
            },
            "dataset_provenance": "genuine-ember-2018" if is_genuine else "synthetic",
            "hyperparameters": {
                "boosting_type": self.config.BOOSTING_TYPE,
                "objective": self.config.OBJECTIVE,
                "metric": list(self.config.METRIC),
                "learning_rate": self.config.LEARNING_RATE,
                "num_leaves": self.config.NUM_LEAVES,
                "max_depth": self.config.MAX_DEPTH,
                "min_data_in_leaf": self.config.MIN_DATA_IN_LEAF,
                "feature_fraction": self.config.FEATURE_FRACTION,
                "bagging_fraction": self.config.BAGGING_FRACTION,
                "bagging_freq": self.config.BAGGING_FREQ,
                "num_boost_round": self.config.NUM_BOOST_ROUND,
                "early_stopping_rounds": self.config.EARLY_STOPPING_ROUNDS,
                "random_seed": self.config.RANDOM_SEED,
            },
            "provenance_notes": provenance_notes or "Model trained on Stage 1 EMBER 2018 schema dataset",
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model metadata saved to {metadata_path}")
        return model_path, metadata_path


__all__ = [
    "ModelTrainer",
    "TrainingResult",
    "generate_ember_feature_names",
]
