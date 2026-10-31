#!/usr/bin/env python
"""
ThreatLens PE Static Feature Extraction CLI

Extracts structural and statistical features from a Windows PE executable
and prints or saves the verified JSON feature dictionary.
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure threatlens is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threatlens.core.adapter import get_adapter
from threatlens.core.extractor import PEFeatureExtractor, PEParseError
from threatlens.core.schema import validate_feature_schema
from threatlens.utils.logger import get_logger

logger = get_logger("extract_cli")


def main():
    parser = argparse.ArgumentParser(
        description="ThreatLens Static PE Feature Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to Windows PE executable (.exe, .dll, .sys)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save output JSON features (prints to stdout if omitted)",
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="Also output the model-ready numerical feature vector using ThreatLensCoreAdapter",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Perform strict schema validation before output",
    )
    args = parser.parse_args()

    if not args.file_path.exists():
        logger.error(f"Target file not found: {args.file_path}")
        sys.exit(1)

    try:
        extractor = PEFeatureExtractor(args.file_path)
        features = extractor.extract()

        is_valid, errors = validate_feature_schema(features, strict=args.strict)
        if not is_valid:
            logger.warning(f"Schema validation warnings: {errors}")
            if args.strict:
                logger.error("Strict validation failed. Aborting.")
                sys.exit(1)

        result_payload = {
            "file": str(args.file_path),
            "features": features,
        }

        if args.vectorize:
            adapter = get_adapter("threatlens_core")
            vector = adapter.transform(features)
            result_payload["vector"] = {
                "pipeline": adapter.pipeline_name,
                "schema_version": adapter.schema_version,
                "dimension": adapter.feature_dim,
                "feature_names": adapter.feature_names,
                "values": vector.tolist(),
            }

        json_output = json.dumps(result_payload, indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json_output, encoding="utf-8")
            print(f"Features successfully saved to {args.output}")
        else:
            print(json_output)

    except PEParseError as e:
        logger.error(f"PE parsing error for '{args.file_path}': {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
