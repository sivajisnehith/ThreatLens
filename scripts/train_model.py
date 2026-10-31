#!/usr/bin/env python
"""
ThreatLens Genuine EMBER 2018 Model Training and Evaluation Pipeline (Stage 7)

Trains the primary LightGBM classifier on the genuine EMBER 2018 2,351-dimensional schema,
applies early stopping on the validation split, fits Platt calibration on the dedicated
calibration split, recomputes global SHAP importances and threshold postures, and evaluates
on the untouched official test split.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Ensure threatlens is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threatlens.config import CALIBRATION_CONFIG, DATA_CONFIG, MODEL_CONFIG, ModelConfig
from threatlens.explainability.explainer import ShapExplainer
from threatlens.explainability.global_explanations import GlobalExplainer
from threatlens.model.calibration import ProbabilityCalibrator
from threatlens.model.evaluator import ModelEvaluator
from threatlens.model.threshold_analysis import ThresholdEvaluator
from threatlens.model.trainer import ModelTrainer
from threatlens.utils.logger import get_logger

logger = get_logger("train_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="ThreatLens Genuine EMBER 2018 Model Training and Evaluation",
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
        default=DATA_CONFIG.GENUINE_MODELS_DIR,
        help="Output directory for serialized genuine model, calibration, and explainability artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Output directory for evaluation reports and figures (defaults to --models-dir)",
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
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP global explanation calculation for rapid debugging",
    )
    args = parser.parse_args()

    reports_dir = args.reports_dir or args.models_dir
    args.models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("THREATLENS STAGE 7: GENUINE EMBER 2018 RETRAINING & INTEGRATION PIPELINE")
    logger.info("================================================================================")

    # 1. Load data splits
    x_train_path = args.data_dir / "X_train.npy"
    y_train_path = args.data_dir / "y_train.npy"
    x_val_path = args.data_dir / "X_val.npy"
    y_val_path = args.data_dir / "y_val.npy"
    x_cal_path = args.data_dir / "X_cal.npy"
    y_cal_path = args.data_dir / "y_cal.npy"
    x_test_path = args.data_dir / "X_test.npy"
    y_test_path = args.data_dir / "y_test.npy"

    for p in [x_train_path, y_train_path, x_val_path, y_val_path, x_cal_path, y_cal_path, x_test_path, y_test_path]:
        if not p.exists():
            logger.error(f"Missing required dataset split: {p}")
            sys.exit(1)

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(x_val_path)
    y_val = np.load(y_val_path)
    X_cal = np.load(x_cal_path)
    y_cal = np.load(y_cal_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    total_samples = len(X_train) + len(X_val) + len(X_cal) + len(X_test)
    total_bytes = X_train.nbytes + X_val.nbytes + X_cal.nbytes + X_test.nbytes
    mem_mb = total_bytes / (1024 * 1024)

    # Pre-training dataset summary disclosure
    print("\n" + "=" * 80)
    print("PRE-TRAINING DATASET & RESOURCE DISCLOSURE")
    print("=" * 80)
    print(f"Dataset Provenance:       genuine-ember-2018 (Elastic EMBER 2018 benchmark)")
    print(f"Total Labeled Samples:    {total_samples:,}")
    print(f"Feature Dimensions:       {X_train.shape[1]} (Canonical EMBER Feature Version 1)")
    print(f"Memory Allocation:        {mem_mb:.2f} MB")
    print(f"Training Split:           {len(X_train):>7,d} samples (Benign: {np.sum(y_train == 0):>6,d}, Malicious: {np.sum(y_train == 1):>6,d})")
    print(f"Validation Split:         {len(X_val):>7,d} samples (Benign: {np.sum(y_val == 0):>6,d}, Malicious: {np.sum(y_val == 1):>6,d})")
    print(f"Calibration Split:        {len(X_cal):>7,d} samples (Benign: {np.sum(y_cal == 0):>6,d}, Malicious: {np.sum(y_cal == 1):>6,d})")
    print(f"Official Isolated Test:   {len(X_test):>7,d} samples (Benign: {np.sum(y_test == 0):>6,d}, Malicious: {np.sum(y_test == 1):>6,d})")
    print("=" * 80 + "\n")

    # 2. Configure training parameters
    config = ModelConfig(
        LEARNING_RATE=args.learning_rate,
        NUM_LEAVES=args.num_leaves,
        NUM_BOOST_ROUND=args.num_boost_round,
        EARLY_STOPPING_ROUNDS=args.early_stopping,
    )

    trainer = ModelTrainer(config=config, models_dir=args.models_dir)
    train_result = trainer.train(X_train, y_train, X_val, y_val)

    # 3. Save model & metadata with genuine provenance
    model_path, metadata_path = trainer.save_model(
        train_result,
        output_dir=args.models_dir,
        provenance_notes="Genuine EMBER 2018 dataset (Stage 7 integration)",
    )

    # 4. Probability Calibration (Platt Scaling on untouched calibration split)
    logger.info("Fitting Platt scaling probability calibrator on untouched calibration split...")
    calibrator = ProbabilityCalibrator(method="sigmoid", models_dir=args.models_dir)
    raw_cal_probs = train_result.booster.predict(X_cal)
    cal_res = calibrator.fit(raw_cal_probs, y_cal)
    cal_path, cal_meta_path = calibrator.save(args.models_dir)
    logger.info(f"Calibrator saved to {cal_path} (Brier: {cal_res.metrics.brier_score_calibrated:.4f}, ECE: {cal_res.metrics.ece_calibrated:.4f})")

    # 5. Evaluate on untouched official test set
    logger.info("Evaluating classifier on untouched official test set (test_features.jsonl)...")
    evaluator = ModelEvaluator(reports_dir=reports_dir)
    metrics = evaluator.evaluate(train_result.booster, X_test, y_test, threshold=args.threshold)
    artifacts = evaluator.generate_evaluation_artifacts(
        train_result.booster,
        X_test,
        y_test,
        feature_names=train_result.feature_names,
        threshold=args.threshold,
    )

    # Calibrated probabilities and metrics on test set
    raw_test_probs = train_result.booster.predict(X_test)
    cal_test_probs = calibrator.predict_proba(raw_test_probs)

    from sklearn.metrics import brier_score_loss, log_loss
    test_brier_raw = float(brier_score_loss(y_test, raw_test_probs))
    test_brier_cal = float(brier_score_loss(y_test, cal_test_probs))
    test_logloss_raw = float(log_loss(y_test, raw_test_probs))
    test_logloss_cal = float(log_loss(y_test, cal_test_probs))

    # 6. Operational Postures & Threshold Analysis on Calibration Split
    logger.info("Computing operational posture thresholds on calibration split...")
    threshold_evaluator = ThresholdEvaluator(reports_dir=args.models_dir)
    threshold_res = threshold_evaluator.evaluate(probabilities=raw_cal_probs, y_true=y_cal)
    threshold_artifacts = threshold_evaluator.generate_full_report(
        probabilities=raw_cal_probs,
        y_true=y_cal,
        output_dir=args.models_dir,
    )
    artifacts.update(threshold_artifacts)

    # 7. SHAP Global Explainability Engine
    if not args.skip_shap:
        logger.info("Computing global SHAP feature importances on evaluation split...")
        try:
            explainer = ShapExplainer(model=train_result.booster, calibrator=calibrator)
            global_engine = GlobalExplainer(explainer=explainer, reports_dir=args.models_dir)
            eval_sample_size = min(500, len(X_val))
            shap_artifacts = global_engine.generate_full_report(
                X=X_val[:eval_sample_size],
                output_dir=args.models_dir,
                top_k=20,
                max_samples=eval_sample_size,
            )
            artifacts.update(shap_artifacts)
        except Exception as e:
            logger.warning(f"Could not complete SHAP global analysis: {e}")

    # Copy / save classification_report.json into models_dir for API loading
    class_report_data = {
        "dataset_provenance": "genuine-ember-2018",
        "threatlens_stage": 7,
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "specificity": metrics.specificity,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "false_positive_rate": metrics.false_positive_rate,
            "false_negative_rate": metrics.false_negative_rate,
            "brier_score_raw": test_brier_raw,
            "brier_score_calibrated": test_brier_cal,
            "log_loss_raw": test_logloss_raw,
            "log_loss_calibrated": test_logloss_cal,
        },
        "confusion_matrix": {
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "true_positives": metrics.true_positives,
        },
    }
    with open(args.models_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(class_report_data, f, indent=2)

    # 8. Log comprehensive results
    print("\n" + "=" * 80)
    print("GENUINE EMBER 2018 MODEL RETRAINING & EVALUATION REPORT")
    print("=" * 80)
    print(f"Model Artifact:           {model_path}")
    print(f"Metadata Artifact:        {metadata_path}")
    print(f"Calibrator Artifact:      {cal_path}")
    print(f"Best Iteration:           {train_result.best_iteration}")
    print(f"Validation Loss:          {train_result.best_score:.4f}")
    print(f"Training Duration:        {train_result.train_duration_sec:.2f}s")
    print("-" * 80)
    print(f"OFFICIAL TEST SET EVALUATION METRICS (N = {len(X_test):,}):")
    print(f"  Accuracy:               {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    print(f"  Precision:              {metrics.precision:.4f} ({metrics.precision * 100:.2f}%)")
    print(f"  Recall (TPR):           {metrics.recall:.4f} ({metrics.recall * 100:.2f}%)")
    print(f"  Specificity (TNR):      {metrics.specificity:.4f} ({metrics.specificity * 100:.2f}%)")
    print(f"  F1-Score:               {metrics.f1:.4f}")
    print(f"  ROC-AUC:                {metrics.roc_auc:.4f}")
    print(f"  PR-AUC:                 {metrics.pr_auc:.4f}")
    print(f"  False Positive Rate:    {metrics.false_positive_rate:.4f} ({metrics.false_positive_rate * 100:.2f}%)")
    print(f"  False Negative Rate:    {metrics.false_negative_rate:.4f} ({metrics.false_negative_rate * 100:.2f}%)")
    print(f"  Raw Brier Score:        {test_brier_raw:.4f}")
    print(f"  Calibrated Brier Score: {test_brier_cal:.4f}")
    print(f"  Raw Log Loss:           {test_logloss_raw:.4f}")
    print(f"  Calibrated Log Loss:    {test_logloss_cal:.4f}")
    print("-" * 80)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives (TN):    {metrics.true_negatives:>6d}  (Correct Benign)")
    print(f"  False Positives (FP):   {metrics.false_positives:>6d}  (False Alarm)")
    print(f"  False Negatives (FN):   {metrics.false_negatives:>6d}  (Missed Malware - CRITICAL)")
    print(f"  True Positives (TP):    {metrics.true_positives:>6d}  (Correct Malware Caught)")
    print("-" * 80)
    print("OPERATIONAL POSTURES (from Calibration Split):")
    for key, posture in threshold_res.postures.items():
        m = posture.metrics if isinstance(posture.metrics, dict) else posture.metrics.to_dict()
        fpr = m.get("false_positive_rate", 0.0) * 100
        rec = m.get("recall", 0.0) * 100
        print(f"  {posture.name:<28}: threshold={posture.threshold:.2f}, FPR={fpr:.2f}%, Recall={rec:.2f}%, Target={posture.target_metric}")
    print("-" * 80)
    print("GENERATED ARTIFACTS:")
    for name, path in artifacts.items():
        print(f"  {name:<25}: {path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
