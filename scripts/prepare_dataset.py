#!/usr/bin/env python
"""
ThreatLens Stage 1: Dataset Preparation Pipeline

Orchestrates the complete dataset preparation workflow:
1. Load EMBER 2018 dataset (or generate synthetic for testing)
2. Validate dataset integrity
3. Split into Train/Val/Cal/Test splits
4. Fit preprocessing pipeline on training data
5. Transform all splits
6. Save artifacts and metadata
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure threatlens is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threatlens.config import DATA_CONFIG, SPLIT_CONFIG
from threatlens.dataset.ember_loader import EmberLoader
from threatlens.dataset.split import split_dataset
from threatlens.dataset.validator import validate_dataset
from threatlens.preprocessing.metadata import MetadataManager
from threatlens.preprocessing.pipeline import EmberPreprocessor
from threatlens.utils.logger import get_logger

logger = get_logger("prepare_dataset")


def main():
    parser = argparse.ArgumentParser(
        description="ThreatLens Stage 1: Dataset Preparation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_CONFIG.RAW_DATA_DIR,
        help="Directory containing EMBER dataset files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to load (for quick testing)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip EMBER download, use existing or synthetic data",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of EMBER dataset",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=4,
        default=[SPLIT_CONFIG.TRAIN_RATIO, SPLIT_CONFIG.VAL_RATIO,
                 SPLIT_CONFIG.CAL_RATIO, SPLIT_CONFIG.TEST_RATIO],
        metavar=("TRAIN", "VAL", "CAL", "TEST"),
        help="Split ratios (must sum to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_CONFIG.RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing pipeline fitting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_CONFIG.PROCESSED_DATA_DIR,
        help="Output directory for processed splits",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger("threatlens").setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ThreatLens Stage 1: Dataset Preparation")
    logger.info("=" * 60)

    try:
        # Step 1: Load or generate data
        logger.info("Step 1/6: Loading EMBER dataset...")
        loader = EmberLoader(data_dir=args.data_dir, random_seed=args.seed)

        if not args.no_download:
            if args.force_download or not loader._check_files_exist():
                loader.download_dataset()

        # Step 1: Load training data and isolated official test data
        logger.info("Step 1/6: Loading EMBER training partition...")
        X_train_raw, y_train_raw = loader.load_or_generate(
            split="train",
            max_samples=args.max_samples,
            generate_if_missing=True,
        )
        logger.info(f"Loaded training partition: {len(X_train_raw)} samples (Benign: {np.sum(y_train_raw == 0)}, Malicious: {np.sum(y_train_raw == 1)})")

        logger.info("Loading official isolated test partition (test_features.jsonl)...")
        test_samples_limit = (args.max_samples // 4) if args.max_samples else None
        X_test_raw, y_test_raw = loader.load_or_generate(
            split="test",
            max_samples=test_samples_limit,
            generate_if_missing=True,
        )
        logger.info(f"Loaded official test partition: {len(X_test_raw)} samples (Benign: {np.sum(y_test_raw == 0)}, Malicious: {np.sum(y_test_raw == 1)})")

        # Step 2: Validate dataset integrity
        if not args.skip_validation:
            logger.info("Step 2/6: Validating training dataset integrity...")
            validation_report = validate_dataset(X_train_raw, y_train_raw)
            if not validation_report.is_valid:
                logger.error("Dataset validation FAILED:")
                for err in validation_report.errors:
                    logger.error(f"  - {err}")
                if not args.max_samples:
                    sys.exit(1)
            else:
                logger.info("Dataset validation PASSED")
        else:
            logger.info("Step 2/6: Skipping validation")
            validation_report = None

        # Step 3: Split training partition into Train, Validation, Calibration (Zero Leakage)
        logger.info("Step 3/6: Partitioning training data into Train, Val, and Calibration splits...")
        from threatlens.dataset.split import DatasetSplits
        from sklearn.model_selection import train_test_split

        # Split ratios for training partition (Train ~70%, Val ~15%, Cal ~15%)
        val_cal_ratio = 0.30
        X_train, X_valcal, y_train, y_valcal = train_test_split(
            X_train_raw,
            y_train_raw,
            test_size=val_cal_ratio,
            random_state=args.seed,
            stratify=y_train_raw,
        )

        X_val, X_cal, y_val, y_cal = train_test_split(
            X_valcal,
            y_valcal,
            test_size=0.50,
            random_state=args.seed,
            stratify=y_valcal,
        )

        # Official test partition remains strictly isolated
        X_test = X_test_raw
        y_test = y_test_raw

        splits = DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
        )

        logger.info("Split summary:")
        for name, info in splits.summary().items():
            logger.info(
                f"  {name:12s}: {info['samples']:>7d} samples "
                f"(benign: {info['benign_count']:>6d}, malicious: {info['malicious_count']:>6d})"
            )

        # Step 4: Fit preprocessing on training data only
        preprocessor = None
        if not args.skip_preprocessing:
            logger.info("Step 4/6: Fitting preprocessing pipeline...")
            preprocessor = EmberPreprocessor()
            preprocessor.fit(splits.X_train)

            logger.info("Transforming all splits...")
            splits.X_train = preprocessor.transform(splits.X_train)
            splits.X_val = preprocessor.transform(splits.X_val)
            splits.X_cal = preprocessor.transform(splits.X_cal)
            splits.X_test = preprocessor.transform(splits.X_test)

            # Save fitted preprocessor
            preprocessor.save()
            logger.info("Preprocessor saved to artifacts/preprocessor.joblib")
        else:
            logger.info("Step 4/6: Skipping preprocessing")

        # Step 5: Save processed splits
        logger.info("Step 5/6: Saving processed splits...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "X_train.npy", splits.X_train)
        np.save(args.output_dir / "y_train.npy", splits.y_train)
        np.save(args.output_dir / "X_val.npy", splits.X_val)
        np.save(args.output_dir / "y_val.npy", splits.y_val)
        np.save(args.output_dir / "X_cal.npy", splits.X_cal)
        np.save(args.output_dir / "y_cal.npy", splits.y_cal)
        np.save(args.output_dir / "X_test.npy", splits.X_test)
        np.save(args.output_dir / "y_test.npy", splits.y_test)
        logger.info(f"Saved split arrays to {args.output_dir}")

        # Step 6: Save metadata
        logger.info("Step 6/6: Saving metadata...")
        metadata_mgr = MetadataManager()
        if validation_report:
            metadata = metadata_mgr.create_metadata_summary(validation_report, splits)
            metadata_mgr.save_metadata(metadata)

        logger.info("=" * 60)
        logger.info("Stage 1 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed data: {args.output_dir}")
        logger.info(f"Preprocessor: {DATA_CONFIG.ARTIFACTS_DIR}/preprocessor.joblib")
        logger.info(f"Metadata: {DATA_CONFIG.ARTIFACTS_DIR}/dataset_metadata.json")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()