#!/usr/bin/env python
"""
ThreatLens Stage 3: LightGBM Model Training and Evaluation Pipeline

Trains the primary LightGBM classifier on the EMBER 2018 2,351-dimensional schema,
applies early stopping on the validation split, evaluates on the untouched test split,
and exports all serialized models and evaluation reports.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Ensure threatlens is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threatlens.config import DATA_CONFIG, MODEL_CONFIG, ModelConfig
from threatlens.model.evaluator import ModelEvaluator
from threatlens.model.trainer import ModelTrainer
from threatlens.utils.logger import get_logger

logger = get_logger("train_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="ThreatLens Stage 3: LightGBM Model Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_CONFIG.PROCESSED_DATA_DIR,
        help="Directory containing processed split .npy files (X_train, y_train, etc.)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DATA_CONFIG.MODELS_DIR,
        help="Output directory for serialized model and metadata",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DATA_CONFIG.REPORTS_DIR,
        help="Output directory for evaluation reports and figures",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=MODEL_CONFIG.LEARNING_RATE,
        help="LightGBM learning rate",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=MODEL_CONFIG.NUM_LEAVES,
        help="LightGBM maximum leaves per tree",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=MODEL_CONFIG.NUM_BOOST_ROUND,
        help="Maximum boosting iterations",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=MODEL_CONFIG.EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds on validation loss",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for malicious classification",
    )
    args = parser.parse_args()

    logger.info("================================================================================")
    logger.info("THREATLENS STAGE 3: LIGHTGBM CLASSIFICATION ENGINE TRAINING PIPELINE")
    logger.info("================================================================================")

    # 1. Load data splits
    x_train_path = args.data_dir / "X_train.npy"
    y_train_path = args.data_dir / "y_train.npy"
    x_val_path = args.data_dir / "X_val.npy"
    y_val_path = args.data_dir / "y_val.npy"
    x_test_path = args.data_dir / "X_test.npy"
    y_test_path = args.data_dir / "y_test.npy"

    for p in [x_train_path, y_train_path, x_val_path, y_val_path, x_test_path, y_test_path]:
        if not p.exists():
            logger.error(f"Missing required dataset split: {p}")
            sys.exit(1)

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(x_val_path)
    y_val = np.load(y_val_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    logger.info(f"Loaded training split:   X={X_train.shape}, y={y_train.shape} (Benign: {np.sum(y_train == 0)}, Malicious: {np.sum(y_train == 1)})")
    logger.info(f"Loaded validation split: X={X_val.shape}, y={y_val.shape} (Benign: {np.sum(y_val == 0)}, Malicious: {np.sum(y_val == 1)})")
    logger.info(f"Loaded test split:       X={X_test.shape}, y={y_test.shape} (Benign: {np.sum(y_test == 0)}, Malicious: {np.sum(y_test == 1)})")

    # 2. Configure training parameters
    config = ModelConfig(
        LEARNING_RATE=args.learning_rate,
        NUM_LEAVES=args.num_leaves,
        NUM_BOOST_ROUND=args.num_boost_round,
        EARLY_STOPPING_ROUNDS=args.early_stopping,
    )

    trainer = ModelTrainer(config=config, models_dir=args.models_dir)
    train_result = trainer.train(X_train, y_train, X_val, y_val)

    # 3. Save model & metadata
    model_path, metadata_path = trainer.save_model(
        train_result,
        output_dir=args.models_dir,
        provenance_notes="Stage 3 LightGBM binary classifier trained on Stage 1 EMBER 2018 schema dataset.",
    )

    # 4. Evaluate on untouched test set
    evaluator = ModelEvaluator(reports_dir=args.reports_dir)
    metrics = evaluator.evaluate(train_result.booster, X_test, y_test, threshold=args.threshold)
    artifacts = evaluator.generate_evaluation_artifacts(
        train_result.booster,
        X_test,
        y_test,
        feature_names=train_result.feature_names,
        threshold=args.threshold,
    )

    # 5. Log comprehensive results
    print("\n" + "=" * 80)
    print("STAGE 3 MODEL TRAINING & EVALUATION REPORT")
    print("=" * 80)
    print(f"Model File:               {model_path}")
    print(f"Metadata File:            {metadata_path}")
    print(f"Best Iteration:           {train_result.best_iteration}")
    print(f"Validation Loss:          {train_result.best_score:.4f}")
    print(f"Training Duration:        {train_result.train_duration_sec:.2f}s")
    print("-" * 80)
    print("TEST SET EVALUATION METRICS (Untouched Test Split, N = 240):")
    print(f"  Accuracy:               {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    print(f"  Precision:              {metrics.precision:.4f} ({metrics.precision * 100:.2f}%)")
    print(f"  Recall (TPR):           {metrics.recall:.4f} ({metrics.recall * 100:.2f}%)")
    print(f"  Specificity (TNR):      {metrics.specificity:.4f} ({metrics.specificity * 100:.2f}%)")
    print(f"  F1-Score:               {metrics.f1:.4f}")
    print(f"  ROC-AUC:                {metrics.roc_auc:.4f}")
    print(f"  PR-AUC:                 {metrics.pr_auc:.4f}")
    print(f"  False Positive Rate:    {metrics.false_positive_rate:.4f} ({metrics.false_positive_rate * 100:.2f}%)")
    print(f"  False Negative Rate:    {metrics.false_negative_rate:.4f} ({metrics.false_negative_rate * 100:.2f}%)")
    print("-" * 80)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives (TN):    {metrics.true_negatives:4d}  (Correct Benign)")
    print(f"  False Positives (FP):   {metrics.false_positives:4d}  (False Alarm)")
    print(f"  False Negatives (FN):   {metrics.false_negatives:4d}  (Missed Malware - CRITICAL)")
    print(f"  True Positives (TP):    {metrics.true_positives:4d}  (Correct Malware Caught)")
    print("-" * 80)
    print("GENERATED ARTIFACTS:")
    for name, path in artifacts.items():
        print(f"  {name:<25}: {path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
