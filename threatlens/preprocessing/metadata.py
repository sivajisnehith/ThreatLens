"""
Dataset Metadata and Config Manager

Handles exporting, saving, and loading metadata reports for dataset provenance,
feature schema specs, split ratios, preprocessing configs, and validation summaries.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from threatlens.config import (
    DATA_CONFIG,
    FEATURE_GROUPS,
    PREPROCESSING_CONFIG,
    SPLIT_CONFIG,
    EmberFeatureGroups,
)
from threatlens.dataset.split import DatasetSplits
from threatlens.dataset.validator import ValidationReport
from threatlens.utils.logger import get_logger

logger = get_logger("metadata")


class MetadataManager:
    """Manager for dataset metadata generation and serialization."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize MetadataManager.

        Args:
            artifacts_dir: Directory to store JSON metadata files
        """
        self.artifacts_dir = artifacts_dir or DATA_CONFIG.ARTIFACTS_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def generate_feature_schema(self) -> Dict[str, Any]:
        """Generate EMBER feature schema definition."""
        groups = {}
        for name in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(name)
            groups[name] = {
                "start_idx": start,
                "end_idx": end,
                "feature_count": end - start,
            }

        return {
            "dataset": "EMBER 2018",
            "total_features": EmberFeatureGroups.TOTAL_FEATURES,
            "feature_groups": groups,
        }

    def create_metadata_summary(
        self,
        validation_report: ValidationReport,
        splits: DatasetSplits,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create complete metadata summary.

        Args:
            validation_report: Report from DatasetValidator
            splits: DatasetSplits container object
            notes: Optional human-readable notes

        Returns:
            Dictionary with complete metadata
        """
        schema = self.generate_feature_schema()

        metadata = {
            "threatlens_stage": 1,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": SPLIT_CONFIG.RANDOM_SEED,
            "feature_schema": schema,
            "split_ratios": {
                "train": SPLIT_CONFIG.TRAIN_RATIO,
                "validation": SPLIT_CONFIG.VAL_RATIO,
                "calibration": SPLIT_CONFIG.CAL_RATIO,
                "test": SPLIT_CONFIG.TEST_RATIO,
            },
            "split_counts": splits.summary(),
            "validation_report": validation_report.to_dict(),
            "preprocessing_config": {
                "impute_strategy": PREPROCESSING_CONFIG.IMPUTE_STRATEGY,
                "scaler_type": PREPROCESSING_CONFIG.SCALER_TYPE,
                "with_centering": PREPROCESSING_CONFIG.WITH_CENTERING,
                "with_scaling": PREPROCESSING_CONFIG.WITH_SCALING,
            },
            "notes": notes or "Stage 1 Dataset Preparation Completed",
        }

        return metadata

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "dataset_metadata.json",
    ) -> Path:
        """
        Save metadata dictionary as JSON file.

        Args:
            metadata: Metadata dictionary
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        save_path = self.artifacts_dir / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset metadata to {save_path}")
        return save_path

    def load_metadata(self, filename: str = "dataset_metadata.json") -> Dict[str, Any]:
        """Load dataset metadata from JSON file."""
        load_path = self.artifacts_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata


__all__ = ["MetadataManager"]