#!/usr/bin/env python
"""
ThreatLens Stage 3: Malware Classifier CLI Inference

Loads the trained LightGBM model and performs validated predictions
on 2,351-dimensional EMBER feature vectors.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Ensure threatlens is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threatlens.config import DATA_CONFIG, MODEL_CONFIG
from threatlens.model.classifier import MalwareClassifier
from threatlens.utils.logger import get_logger

logger = get_logger("predict_cli")


def main():
    parser = argparse.ArgumentParser(
        description="ThreatLens Malware Classifier Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to feature vector file (.npy array of shape (2351,) or (N, 2351), or .json file)",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for malicious classification",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save output prediction JSON (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    classifier = MalwareClassifier(model_path=args.model_path, metadata_path=args.metadata_path)

    # Load input
    if args.input_path.suffix.lower() == ".npy":
        features = np.load(args.input_path)
    elif args.input_path.suffix.lower() == ".json":
        with open(args.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "features" in data:
            features = data["features"]
        elif isinstance(data, dict) and "values" in data:
            features = data["values"]
        else:
            features = data
    else:
        logger.error(f"Unsupported file format: {args.input_path.suffix}. Expected .npy or .json")
        sys.exit(1)

    if isinstance(features, np.ndarray) and features.ndim == 2 and len(features) > 1:
        # Batch inference
        batch_result = classifier.predict_batch(features, threshold=args.threshold)
        output_payload = batch_result.to_dict()
    else:
        # Single sample inference
        sample = features[0] if isinstance(features, np.ndarray) and features.ndim == 2 else features
        pred_result = classifier.predict_single(sample, threshold=args.threshold)
        output_payload = pred_result.to_dict()

    json_str = json.dumps(output_payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Predictions saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
