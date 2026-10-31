"""
EMBER Dataset Validation Module

Comprehensive validation suite for checking EMBER dataset integrity, feature dimensions,
missing/infinite values, duplicates, class distributions, and corruption.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from threatlens.config import FEATURE_GROUPS, EmberFeatureGroups
from threatlens.utils.logger import get_logger

logger = get_logger("validator")


@dataclass
class ValidationReport:
    """Dataclass holding complete dataset validation results."""

    is_valid: bool
    total_samples: int
    num_features: int
    valid_features_dim: bool
    label_distribution: Dict[int, int]
    class_imbalance_ratio: float
    missing_value_count: int
    missing_value_percentage: float
    infinite_value_count: int
    duplicate_record_count: int
    corrupted_record_count: int
    invalid_label_count: int
    group_level_stats: Dict[str, Dict[str, float]]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert validation report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_samples": self.total_samples,
            "num_features": self.num_features,
            "valid_features_dim": self.valid_features_dim,
            "label_distribution": self.label_distribution,
            "class_imbalance_ratio": round(self.class_imbalance_ratio, 4),
            "missing_value_count": self.missing_value_count,
            "missing_value_percentage": round(self.missing_value_percentage, 4),
            "infinite_value_count": self.infinite_value_count,
            "duplicate_record_count": self.duplicate_record_count,
            "corrupted_record_count": self.corrupted_record_count,
            "invalid_label_count": self.invalid_label_count,
            "group_level_stats": self.group_level_stats,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DatasetValidator:
    """
    Validator for EMBER dataset features and labels.
    """

    def __init__(
        self,
        expected_feature_dim: int = EmberFeatureGroups.TOTAL_FEATURES,
        max_missing_pct: float = 0.05,
        max_corrupted_pct: float = 0.02,
    ):
        """
        Initialize the validator.

        Args:
            expected_feature_dim: Expected number of features (default: 2351)
            max_missing_pct: Maximum allowed proportion of missing/NaN values
            max_corrupted_pct: Maximum allowed proportion of corrupted records
        """
        self.expected_feature_dim = expected_feature_dim
        self.max_missing_pct = max_missing_pct
        self.max_corrupted_pct = max_corrupted_pct

    def validate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        check_duplicates: bool = True,
    ) -> ValidationReport:
        """
        Run complete validation checks on features and labels.

        Args:
            features: 2D numpy array of shape (N, D)
            labels: 1D numpy array of shape (N,)
            check_duplicates: Whether to perform duplicate detection (can be slow on huge datasets)

        Returns:
            ValidationReport instance
        """
        logger.info(f"Starting dataset validation for {features.shape[0]} samples...")

        errors = []
        warnings = []

        # 1. Shape and Dimension Checks
        if features.ndim != 2:
            errors.append(f"Features must be a 2D array, got {features.ndim}D array")
            total_samples, num_features = (0, 0)
        else:
            total_samples, num_features = features.shape

        if labels.ndim != 1:
            errors.append(f"Labels must be a 1D array, got {labels.ndim}D array")
        elif len(labels) != total_samples:
            errors.append(f"Sample count mismatch: {total_samples} features vs {len(labels)} labels")

        valid_features_dim = (num_features == self.expected_feature_dim)
        if not valid_features_dim:
            errors.append(
                f"Feature dimension mismatch: expected {self.expected_feature_dim}, got {num_features}"
            )

        if total_samples == 0:
            errors.append("Dataset is empty")
            return ValidationReport(
                is_valid=False,
                total_samples=0,
                num_features=num_features,
                valid_features_dim=valid_features_dim,
                label_distribution={},
                class_imbalance_ratio=0.0,
                missing_value_count=0,
                missing_value_percentage=0.0,
                infinite_value_count=0,
                duplicate_record_count=0,
                corrupted_record_count=0,
                invalid_label_count=0,
                group_level_stats={},
                warnings=warnings,
                errors=errors,
            )

        # 2. Missing and Infinite Values
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)

        missing_value_count = int(np.sum(nan_mask))
        infinite_value_count = int(np.sum(inf_mask))

        total_cells = total_samples * num_features
        missing_value_percentage = (missing_value_count / total_cells) * 100.0 if total_cells > 0 else 0.0

        if missing_value_percentage > (self.max_missing_pct * 100.0):
            warnings.append(
                f"High missing value rate: {missing_value_percentage:.2f}% (thresh: {self.max_missing_pct*100}%)"
            )

        if infinite_value_count > 0:
            warnings.append(f"Found {infinite_value_count} infinite values in feature matrix")

        # 3. Label Validation & Class Imbalance
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_distribution = {int(k): int(v) for k, v in zip(unique_labels, label_counts)}

        # Check valid binary labels (0 = benign, 1 = malicious)
        invalid_labels = [l for l in unique_labels if l not in [0, 1]]
        invalid_label_count = sum(label_distribution.get(l, 0) for l in invalid_labels)

        if invalid_label_count > 0:
            errors.append(f"Found {invalid_label_count} samples with invalid labels: {invalid_labels}")

        # Class imbalance calculation
        count_0 = label_distribution.get(0, 0)
        count_1 = label_distribution.get(1, 0)
        min_class = min(count_0, count_1)
        max_class = max(count_0, count_1)
        class_imbalance_ratio = (max_class / min_class) if min_class > 0 else float("inf")

        if class_imbalance_ratio > 3.0:
            warnings.append(f"Significant class imbalance detected: ratio {class_imbalance_ratio:.2f}:1")

        # 4. Corrupted Records Detection
        # A record is corrupted if all feature values are zero, NaN, or infinite
        row_nan_inf = np.isnan(features).all(axis=1) | np.isinf(features).all(axis=1)
        row_all_zeros = (features == 0).all(axis=1)
        corrupted_mask = row_nan_inf | row_all_zeros
        corrupted_record_count = int(np.sum(corrupted_mask))

        corrupted_pct = corrupted_record_count / total_samples
        if corrupted_pct > self.max_corrupted_pct:
            warnings.append(f"Corrupted records ({corrupted_record_count}) exceed threshold ({corrupted_pct*100:.1f}%)")

        # 5. Duplicate Records Detection
        duplicate_record_count = 0
        if check_duplicates and total_samples <= 50000:  # Limit for memory/speed
            try:
                # Find duplicates via hashing rows
                feature_bytes = np.ascontiguousarray(features).view(np.uint8)
                hashes = [hash(row.tobytes()) for row in feature_bytes]
                unique_hashes = len(set(hashes))
                duplicate_record_count = total_samples - unique_hashes
                if duplicate_record_count > 0:
                    warnings.append(f"Found {duplicate_record_count} duplicate feature vectors")
            except Exception as e:
                logger.debug(f"Duplicate check skipped due to error: {e}")

        # 6. Feature Group Level Statistics
        group_level_stats = self._compute_group_stats(features)

        # Final verdict
        is_valid = len(errors) == 0

        report = ValidationReport(
            is_valid=is_valid,
            total_samples=total_samples,
            num_features=num_features,
            valid_features_dim=valid_features_dim,
            label_distribution=label_distribution,
            class_imbalance_ratio=class_imbalance_ratio,
            missing_value_count=missing_value_count,
            missing_value_percentage=missing_value_percentage,
            infinite_value_count=infinite_value_count,
            duplicate_record_count=duplicate_record_count,
            corrupted_record_count=corrupted_record_count,
            invalid_label_count=invalid_label_count,
            group_level_stats=group_level_stats,
            warnings=warnings,
            errors=errors,
        )

        logger.info(
            f"Validation complete. Verdict: {'VALID' if is_valid else 'INVALID'} "
            f"({len(errors)} errors, {len(warnings)} warnings)"
        )
        return report

    def _compute_group_stats(self, features: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute mean, std, min, max across EMBER feature groups."""
        stats = {}
        for group_name in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(group_name)
            if end <= features.shape[1]:
                group_feats = features[:, start:end]
                stats[group_name] = {
                    "mean": float(np.nanmean(group_feats)),
                    "std": float(np.nanstd(group_feats)),
                    "min": float(np.nanmin(group_feats)),
                    "max": float(np.nanmax(group_feats)),
                    "nan_count": int(np.isnan(group_feats).sum()),
                }
        return stats


def validate_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    expected_dim: int = EmberFeatureGroups.TOTAL_FEATURES,
) -> ValidationReport:
    """Convenience validation function."""
    validator = DatasetValidator(expected_feature_dim=expected_dim)
    return validator.validate(features, labels)


__all__ = [
    "DatasetValidator",
    "ValidationReport",
    "validate_dataset",
]