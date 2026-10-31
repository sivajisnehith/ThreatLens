"""
Dataset Split Module

Handles dataset partitioning into Train, Validation, Calibration, and Test splits
ensuring zero data leakage and maintaining class balance via stratified splitting.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from threatlens.config import SPLIT_CONFIG, SplitConfig
from threatlens.utils.logger import get_logger

logger = get_logger("split")


@dataclass
class DatasetSplits:
    """Dataclass holding separated dataset splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def summary(self) -> Dict[str, Dict[str, int]]:
        """Return summary of split shapes and label counts."""
        splits = {
            "train": (self.X_train, self.y_train),
            "validation": (self.X_val, self.y_val),
            "calibration": (self.X_cal, self.y_cal),
            "test": (self.X_test, self.y_test),
        }

        summary_dict = {}
        for name, (X, y) in splits.items():
            unique, counts = np.unique(y, return_counts=True)
            dist = {int(k): int(v) for k, v in zip(unique, counts)}
            summary_dict[name] = {
                "samples": len(X),
                "features": X.shape[1] if X.ndim > 1 else 0,
                "benign_count": dist.get(0, 0),
                "malicious_count": dist.get(1, 0),
            }
        return summary_dict


class DataSplitter:
    """
    Data splitter to split feature matrix X and labels y into
    Train, Validation, Calibration, and Test splits without leakage.
    """

    def __init__(self, config: SplitConfig = SPLIT_CONFIG):
        """
        Initialize data splitter.

        Args:
            config: SplitConfig instance holding ratios and random seed
        """
        self.config = config

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> DatasetSplits:
        """
        Split features and labels into Train, Val, Cal, Test.

        Order of operations:
        1. Extract Test set (test_ratio)
        2. From remaining, extract Calibration set (cal_ratio / remaining_ratio)
        3. From remaining, extract Validation set (val_ratio / remaining_ratio)
        4. Remaining is Training set.

        Args:
            X: Feature array of shape (N, D)
            y: Label array of shape (N,)

        Returns:
            DatasetSplits container object
        """
        logger.info(f"Splitting dataset of shape {X.shape} into 4 splits...")

        seed = self.config.RANDOM_SEED
        stratify_flag = self.config.STRATIFY

        # Calculate proportions
        test_ratio = self.config.TEST_RATIO
        cal_ratio = self.config.CAL_RATIO
        val_ratio = self.config.VAL_RATIO
        train_ratio = self.config.TRAIN_RATIO

        # Step 1: Split off Test set
        strat_y = y if stratify_flag else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=seed,
            stratify=strat_y,
        )

        # Remaining ratio = train + val + cal
        remaining_ratio = train_ratio + val_ratio + cal_ratio

        # Step 2: Split off Calibration set from remaining
        cal_relative_ratio = cal_ratio / remaining_ratio
        strat_y_temp = y_temp if stratify_flag else None

        X_temp2, X_cal, y_temp2, y_cal = train_test_split(
            X_temp,
            y_temp,
            test_size=cal_relative_ratio,
            random_state=seed,
            stratify=strat_y_temp,
        )

        # Remaining ratio for train + val
        remaining_ratio2 = train_ratio + val_ratio

        # Step 3: Split off Validation set from remaining
        val_relative_ratio = val_ratio / remaining_ratio2
        strat_y_temp2 = y_temp2 if stratify_flag else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp2,
            y_temp2,
            test_size=val_relative_ratio,
            random_state=seed,
            stratify=strat_y_temp2,
        )

        logger.info(
            f"Splits complete: Train={len(X_train)}, Val={len(X_val)}, "
            f"Cal={len(X_cal)}, Test={len(X_test)}"
        )

        return DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
        )


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    config: SplitConfig = SPLIT_CONFIG,
) -> DatasetSplits:
    """Convenience splitting function."""
    splitter = DataSplitter(config)
    return splitter.split(X, y)


__all__ = [
    "DataSplitter",
    "DatasetSplits",
    "split_dataset",
]