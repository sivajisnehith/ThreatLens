"""
Unit Tests for Configuration Module
"""

import numpy as np
import pytest

from threatlens.config import (
    DATA_CONFIG,
    SPLIT_CONFIG,
    PREPROCESSING_CONFIG,
    LOGGING_CONFIG,
    FEATURE_GROUPS,
    EmberFeatureGroups,
    DataConfig,
    SplitConfig,
    PreprocessingConfig,
    LoggingConfig,
)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_paths(self):
        assert DATA_CONFIG.PROJECT_ROOT.exists()
        assert DATA_CONFIG.DATA_DIR == DATA_CONFIG.PROJECT_ROOT / "data"
        assert DATA_CONFIG.RAW_DATA_DIR == DATA_CONFIG.PROJECT_ROOT / "data" / "raw"
        assert DATA_CONFIG.PROCESSED_DATA_DIR == DATA_CONFIG.PROJECT_ROOT / "data" / "processed"
        assert DATA_CONFIG.ARTIFACTS_DIR == DATA_CONFIG.PROJECT_ROOT / "artifacts"
        assert DATA_CONFIG.LOGS_DIR == DATA_CONFIG.PROJECT_ROOT / "logs"
        assert DATA_CONFIG.REPORTS_DIR == DATA_CONFIG.PROJECT_ROOT / "reports"

    def test_ember_constants(self):
        assert DATA_CONFIG.EMBER_VERSION == "2018"
        assert "ember.elastic.co" in DATA_CONFIG.EMBER_URL
        assert DATA_CONFIG.EMBER_TRAIN_FEATURES == "train_features.jsonl"
        assert DATA_CONFIG.EMBER_TEST_LABELS == "test_labels.csv"


class TestSplitConfig:
    """Tests for SplitConfig."""

    def test_default_ratios_sum_to_one(self):
        total = (SPLIT_CONFIG.TRAIN_RATIO + SPLIT_CONFIG.VAL_RATIO +
                 SPLIT_CONFIG.CAL_RATIO + SPLIT_CONFIG.TEST_RATIO)
        assert abs(total - 1.0) < 1e-6

    def test_default_ratios(self):
        assert SPLIT_CONFIG.TRAIN_RATIO == 0.60
        assert SPLIT_CONFIG.VAL_RATIO == 0.15
        assert SPLIT_CONFIG.CAL_RATIO == 0.15
        assert SPLIT_CONFIG.TEST_RATIO == 0.10

    def test_random_seed(self):
        assert SPLIT_CONFIG.RANDOM_SEED == 42

    def test_stratify_enabled(self):
        assert SPLIT_CONFIG.STRATIFY is True

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SplitConfig(TRAIN_RATIO=0.5, VAL_RATIO=0.2, CAL_RATIO=0.2, TEST_RATIO=0.2)

    def test_custom_config(self):
        custom = SplitConfig(TRAIN_RATIO=0.7, VAL_RATIO=0.1, CAL_RATIO=0.1, TEST_RATIO=0.1)
        assert custom.TRAIN_RATIO == 0.7


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_defaults(self):
        assert PREPROCESSING_CONFIG.IMPUTE_STRATEGY == "median"
        assert PREPROCESSING_CONFIG.SCALER_TYPE == "standard"
        assert PREPROCESSING_CONFIG.WITH_CENTERING is True
        assert PREPROCESSING_CONFIG.WITH_SCALING is True
        assert PREPROCESSING_CONFIG.RANDOM_SEED == 42


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_defaults(self):
        assert LOGGING_CONFIG.LEVEL == "INFO"
        assert "%(asctime)s" in LOGGING_CONFIG.FORMAT
        assert LOGGING_CONFIG.LOG_FILE == "threatlens.log"


class TestEmberFeatureGroups:
    """Tests for EmberFeatureGroups (also in test_dataset_validation.py but repeated here for config)."""

    def test_total_features(self):
        assert FEATURE_GROUPS.TOTAL_FEATURES == 2351

    def test_group_indices(self):
        # Use class methods instead of instance attribute
        start, end = EmberFeatureGroups.get_group_indices("histogram")
        assert (start, end) == (0, 256)
        start, end = EmberFeatureGroups.get_group_indices("byteentropy")
        assert (start, end) == (256, 512)
        start, end = EmberFeatureGroups.get_group_indices("strings")
        assert (start, end) == (512, 768)
        start, end = EmberFeatureGroups.get_group_indices("general")
        assert (start, end) == (768, 778)
        start, end = EmberFeatureGroups.get_group_indices("header")
        assert (start, end) == (778, 839)
        start, end = EmberFeatureGroups.get_group_indices("section")
        assert (start, end) == (839, 1035)
        start, end = EmberFeatureGroups.get_group_indices("imports")
        assert (start, end) == (1035, 2283)
        start, end = EmberFeatureGroups.get_group_indices("exports")
        assert (start, end) == (2283, 2351)

    def test_get_group_indices(self):
        start, end = EmberFeatureGroups.get_group_indices("histogram")
        assert start == 0
        assert end == 256

        start, end = EmberFeatureGroups.get_group_indices("exports")
        assert start == 2283
        assert end == 2351

    def test_get_all_group_names(self):
        names = EmberFeatureGroups.get_all_group_names()
        assert len(names) == 8
        assert names == ["histogram", "byteentropy", "strings", "general",
                         "header", "section", "imports", "exports"]

    def test_validate_feature_dim(self):
        assert EmberFeatureGroups.validate_feature_dim(2351) is True
        assert EmberFeatureGroups.validate_feature_dim(2000) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])