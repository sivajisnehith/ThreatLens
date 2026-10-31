"""
Unit Tests for Dataset Validation Module
"""

import numpy as np
import pytest

from threatlens.config import FEATURE_GROUPS, EmberFeatureGroups
from threatlens.dataset.validator import DatasetValidator, ValidationReport, validate_dataset


class TestEmberFeatureGroups:
    """Tests for EMBER feature group definitions."""

    def test_total_features(self):
        assert EmberFeatureGroups.TOTAL_FEATURES == 2351

    def test_group_names(self):
        groups = EmberFeatureGroups.get_all_group_names()
        expected = ["histogram", "byteentropy", "strings", "general",
                    "header", "section", "imports", "exports"]
        assert groups == expected

    def test_group_indices(self):
        # Test first group
        start, end = EmberFeatureGroups.get_group_indices("histogram")
        assert start == 0
        assert end == 256

        # Test last group
        start, end = EmberFeatureGroups.get_group_indices("exports")
        assert start == 2223
        assert end == 2351

        # Test unknown group returns (0, 0)
        start, end = EmberFeatureGroups.get_group_indices("unknown")
        assert start == 0
        assert end == 0

    def test_validate_dim(self):
        assert EmberFeatureGroups.validate_feature_dim(2351) is True
        assert EmberFeatureGroups.validate_feature_dim(2350) is False
        assert EmberFeatureGroups.validate_feature_dim(2000) is False


class TestDatasetValidator:
    """Tests for DatasetValidator class."""

    def setup_method(self):
        """Create test fixtures."""
        self.validator = DatasetValidator(expected_feature_dim=2351)
        np.random.seed(42)

    def create_valid_data(self, n_samples=100, n_features=2351):
        """Create valid test data."""
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples, dtype=np.int32)
        return X, y

    def test_validate_valid_data(self):
        """Test validation passes on clean data."""
        X, y = self.create_valid_data()
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.is_valid is True
        assert report.total_samples == 100
        assert report.num_features == 2351
        assert report.valid_features_dim is True
        assert report.missing_value_count == 0
        assert report.infinite_value_count == 0
        assert report.corrupted_record_count == 0
        assert report.invalid_label_count == 0
        assert len(report.errors) == 0

    def test_validate_wrong_dim(self):
        """Test validation fails on wrong feature dimension."""
        X = np.random.randn(50, 2000).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.is_valid is False
        assert report.valid_features_dim is False
        assert any("Feature dimension mismatch" in e for e in report.errors)

    def test_validate_mismatched_samples(self):
        """Test validation fails on sample count mismatch."""
        X = np.random.randn(100, 2351).astype(np.float32)
        y = np.random.randint(0, 2, 90)
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.is_valid is False
        assert any("Sample count mismatch" in e for e in report.errors)

    def test_validate_nan_values(self):
        """Test validation detects NaN values."""
        X, y = self.create_valid_data()
        X[0, 0] = np.nan
        X[5, 100] = np.nan
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.is_valid is True  # NaNs are warnings, not errors
        assert report.missing_value_count == 2
        assert report.missing_value_percentage > 0
        # Add more NaN to exceed default 5% threshold
        # 5% of 235100 = 11755 cells
        X[10:60, 0:250] = np.nan  # Add 50*250 = 12500 NaNs
        report2 = self.validator.validate(X, y, check_duplicates=False)
        assert any("missing" in w.lower() for w in report2.warnings)

    def test_validate_inf_values(self):
        """Test validation detects infinite values."""
        X, y = self.create_valid_data()
        X[0, 0] = np.inf
        X[10, 500] = -np.inf
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.infinite_value_count == 2
        assert any("infinite" in w.lower() for w in report.warnings)

    def test_validate_invalid_labels(self):
        """Test validation fails on invalid labels."""
        X = np.random.randn(50, 2351).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        y[0] = 2  # Invalid label
        y[5] = -1  # Invalid label
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.is_valid is False
        assert report.invalid_label_count == 2
        assert any("invalid label" in e.lower() for e in report.errors)

    def test_validate_empty_dataset(self):
        """Test validation fails on empty dataset."""
        X = np.empty((0, 2351), dtype=np.float32)
        y = np.empty(0, dtype=np.int32)
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.is_valid is False
        assert any("empty" in e.lower() for e in report.errors)

    def test_validate_corrupted_records(self):
        """Test validation detects all-zero/NaN rows."""
        X, y = self.create_valid_data(200)
        # Add corrupted rows
        X[0, :] = 0  # All zeros
        X[1, :] = np.nan  # All NaN
        X[2, :] = np.inf  # All Inf
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.corrupted_record_count == 3
        # Should be warning if under threshold (default 2%)
        if report.corrupted_record_count / 200 > 0.02:
            assert any("corrupted" in w.lower() for w in report.warnings)

    def test_validate_duplicate_records(self):
        """Test duplicate detection works."""
        X = np.random.randn(10, 2351).astype(np.float32)
        X[5] = X[0]  # Duplicate row 0
        X[8] = X[2]  # Duplicate row 2
        y = np.random.randint(0, 2, 10)
        report = self.validator.validate(X, y, check_duplicates=True)

        assert report.duplicate_record_count == 2
        assert any("duplicate" in w.lower() for w in report.warnings)

    def test_validate_class_imbalance(self):
        """Test class imbalance detection."""
        X = np.random.randn(1000, 2351).astype(np.float32)
        y = np.zeros(1000, dtype=np.int32)
        y[:10] = 1  # Only 10 malicious, 990 benign -> 99:1 ratio
        report = self.validator.validate(X, y, check_duplicates=False)

        assert report.class_imbalance_ratio == 99.0
        assert any("imbalance" in w.lower() for w in report.warnings)

    def test_validate_group_stats(self):
        """Test feature group statistics are computed."""
        X, y = self.create_valid_data(50)
        report = self.validator.validate(X, y, check_duplicates=False)

        assert "group_level_stats" in report.to_dict()
        groups = report.group_level_stats
        assert len(groups) == 8  # 8 feature groups

        for name in FEATURE_GROUPS.get_all_group_names():
            assert name in groups
            assert "mean" in groups[name]
            assert "std" in groups[name]
            assert "min" in groups[name]
            assert "max" in groups[name]

    def test_report_to_dict(self):
        """Test ValidationReport.to_dict() serialization."""
        X, y = self.create_valid_data(10)
        report = self.validator.validate(X, y, check_duplicates=False)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["is_valid"] is True
        assert d["total_samples"] == 10
        assert d["num_features"] == 2351
        assert "label_distribution" in d
        assert "warnings" in d
        assert "errors" in d


class TestConvenienceFunction:
    """Test the validate_dataset convenience function."""

    def test_validate_dataset_function(self):
        X = np.random.randn(20, 2351).astype(np.float32)
        y = np.random.randint(0, 2, 20)
        report = validate_dataset(X, y)

        assert isinstance(report, ValidationReport)
        assert report.is_valid is True


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_high_nan_threshold(self):
        """Test behavior when missing values exceed threshold."""
        validator = DatasetValidator(expected_feature_dim=100, max_missing_pct=0.01)
        X = np.random.randn(100, 100).astype(np.float32)
        # Introduce 5% NaN
        X[:5, :] = np.nan
        y = np.random.randint(0, 2, 100)
        report = validator.validate(X, y, check_duplicates=False)

        assert report.missing_value_percentage == 5.0
        assert any("missing" in w.lower() for w in report.warnings)

    def test_dtype_preservation(self):
        """Test validator handles different dtypes."""
        X = np.random.randn(10, 2351).astype(np.float64)
        y = np.random.randint(0, 2, 10).astype(np.int64)
        report = validate_dataset(X, y)
        assert report.is_valid is True

    def test_integer_labels_various_types(self):
        """Test label validation with different integer types."""
        X = np.random.randn(10, 2351).astype(np.float32)
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            y = np.array([0, 1] * 5, dtype=dtype)
            report = validate_dataset(X, y)
            assert report.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])