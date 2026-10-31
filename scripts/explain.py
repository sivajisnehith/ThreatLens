#!/usr/bin/env python
"""
ThreatLens Stage 4: SHAP Explainability Engine CLI

Generates global feature importance reports, group aggregations, summary/bar plots,
and granular single-sample prediction explanations using SHAP TreeExplainer.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np

# Ensure threatlens is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threatlens.config import DATA_CONFIG, MODEL_CONFIG
from threatlens.explainability.explainer import ShapExplainer, SingleExplanation
from threatlens.explainability.global_explanations import GlobalExplainer
from threatlens.utils.logger import get_logger

logger = get_logger("explain_cli")


def run_global_explanation(
    data_path: Path,
    output_dir: Path,
    model_path: Optional[Path],
    metadata_path: Optional[Path],
    top_k: int,
    max_samples: int,
) -> None:
    """Execute global SHAP explanation pipeline and write all report artifacts."""
    logger.info("================================================================================")
    logger.info("THREATLENS STAGE 4: GLOBAL SHAP EXPLAINABILITY ANALYSIS")
    logger.info("================================================================================")

    if not data_path.exists():
        logger.error(f"Dataset file not found at: {data_path}")
        sys.exit(1)

    logger.info(f"Loading evaluation dataset from {data_path}...")
    X = np.load(data_path)
    logger.info(f"Loaded dataset with shape {X.shape}")

    logger.info("Initializing ShapExplainer...")
    explainer = ShapExplainer(model=model_path, metadata_path=metadata_path)
    global_engine = GlobalExplainer(explainer=explainer, reports_dir=output_dir)

    artifacts = global_engine.generate_full_report(
        X=X,
        output_dir=output_dir,
        top_k=top_k,
        max_samples=max_samples,
    )

    print("\n" + "=" * 80)
    print("GLOBAL SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"1. Global JSON Report:    {artifacts['json_report']}")
    print(f"2. SHAP Summary Plot:     {artifacts['summary_plot']}")
    print(f"3. SHAP Bar Plot:         {artifacts['bar_plot']}")
    print(f"4. Group Importance Plot: {artifacts['group_plot']}")
    print("=" * 80 + "\n")


def run_local_explanation(
    input_path: Optional[Path],
    data_path: Optional[Path],
    sample_index: int,
    output_path: Optional[Path],
    model_path: Optional[Path],
    metadata_path: Optional[Path],
    top_k: int,
    threshold: float,
) -> None:
    """Execute single prediction local SHAP explanation."""
    logger.info("================================================================================")
    logger.info("THREATLENS STAGE 4: LOCAL PREDICTION SHAP EXPLANATION")
    logger.info("================================================================================")

    # 1. Resolve feature vector
    if input_path is not None:
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        if input_path.suffix.lower() == ".npy":
            raw_data = np.load(input_path)
            feature_vector = raw_data[0] if raw_data.ndim == 2 else raw_data
        elif input_path.suffix.lower() == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                feature_vector = data.get("features", data.get("values", data))
            else:
                feature_vector = data
        else:
            logger.error(f"Unsupported input format: {input_path.suffix}. Expected .npy or .json")
            sys.exit(1)
    elif data_path is not None:
        if not data_path.exists():
            logger.error(f"Dataset split not found: {data_path}")
            sys.exit(1)
        matrix = np.load(data_path)
        if not (0 <= sample_index < len(matrix)):
            logger.error(f"Sample index {sample_index} out of bounds for dataset of length {len(matrix)}")
            sys.exit(1)
        feature_vector = matrix[sample_index]
        logger.info(f"Loaded sample index {sample_index} from {data_path}")
    else:
        logger.error("Must specify either --input-path or --data-path with --sample-index.")
        sys.exit(1)

    explainer = ShapExplainer(model=model_path, metadata_path=metadata_path)
    explanation: SingleExplanation = explainer.explain_single(
        feature_vector=feature_vector,
        top_k=top_k,
        threshold=threshold,
    )

    # Print formatted explanation summary
    print("\n" + "=" * 80)
    print("THREATLENS LOCAL PREDICTION EXPLANATION")
    print("=" * 80)
    print(f"Prediction:          {explanation.prediction_label} (Class {explanation.prediction})")
    print(f"Malware Probability: {explanation.probability * 100:.2f}%")
    print(f"Benign Probability:  {explanation.benign_probability * 100:.2f}%")
    print(f"Raw Model Score:     {explanation.raw_score:.6f} (log-odds)")
    print(f"Baseline (E[f(x)]):  {explanation.expected_value:.6f}")
    print(f"Decision Threshold:  {explanation.threshold:.2f}")
    print(f"Additivity Error:    {explanation.additivity_error:.2e}")
    print("-" * 80)
    print(f"TOP {min(5, len(explanation.top_positive_contributors))} FEATURES INCREASING MALWARE RISK (+ SHAP):")
    for c in explanation.top_positive_contributors[:5]:
        print(f"  [Rank {c.rank}] {c.feature_name:<26} Group: {c.feature_group:<12} SHAP: +{c.shap_value:.6f} | Val: {c.feature_value:.4f}")
        print(f"         Desc: {c.feature_description}")

    print("-" * 80)
    print(f"TOP {min(5, len(explanation.top_negative_contributors))} FEATURES DECREASING MALWARE RISK (- SHAP):")
    for c in explanation.top_negative_contributors[:5]:
        print(f"  [Rank {c.rank}] {c.feature_name:<26} Group: {c.feature_group:<12} SHAP: {c.shap_value:.6f} | Val: {c.feature_value:.4f}")
        print(f"         Desc: {c.feature_description}")
    print("=" * 80)
    print(f"NOTE: {explanation.disclaimer}")
    print("=" * 80 + "\n")

    if output_path is not None:
        explanation.to_json(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="ThreatLens Stage 4: SHAP Explainability Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["global", "local"],
        default="global",
        help="Explanation mode: 'global' for dataset-wide analysis or 'local' for a single sample.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_CONFIG.PROCESSED_DATA_DIR / "X_test.npy",
        help="Path to feature dataset .npy file (used for global analysis or local with --sample-index)",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to single feature file (.npy or .json) for local explanation mode",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample row index to explain when using --mode local with --data-path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_CONFIG.REPORTS_DIR,
        help="Directory to save global explanation reports and plots",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for local explanation JSON",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.MODEL_FILENAME,
        help="Path to trained LightGBM text model",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DATA_CONFIG.MODELS_DIR / MODEL_CONFIG.METADATA_FILENAME,
        help="Path to model_metadata.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top influential features to evaluate/display",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to subsample for global SHAP calculation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for malicious classification",
    )

    args = parser.parse_args()

    if args.mode == "global":
        run_global_explanation(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            top_k=args.top_k,
            max_samples=args.max_samples,
        )
    else:
        run_local_explanation(
            input_path=args.input_path,
            data_path=args.data_path,
            sample_index=args.sample_index,
            output_path=args.output,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            top_k=args.top_k,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
