"""
Unit Tests for Preprocessing Pipeline and Data Splitting
"""

import numpy as np
import pytest

from threatlens.config import DATA_CONFIG, PREPROCESSING_CONFIG, SPLIT_CONFIG
from threatlens.dataset.split import DataSplitter, split_dataset
from threatlens.preprocessing.pipeline import EmberPreprocessor


class TestDataSplitter:
    """Tests for dataset splitting."""

    def setup_method(self):
        np.random.seed(42)

    def create_test_data(self, n_samples=1000, n_features=2351, class_balance=0.5):
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        n_malicious = int(n_samples * class_balance)
        y = np.zeros(n_samples, dtype=np.int32)
        y[:n_malicious] = 1
        np.random.shuffle(y)
        return X, y

    def test_split_default_ratios(self):
        """Test splitting with default ratios."""
        X, y = self.create_test_data(1000)
        splits = split_dataset(X, y)

        assert splits.X_train.shape[0] == 600  # 60%
        assert splits.X_val.shape[0] == 150    # 15%
        assert splits.X_cal.shape[0] == 150    # 15%
        assert splits.X_test.shape[0] == 100   # 10%

        # Check no overlap in indices via data values
        all_X = np.vstack([splits.X_train, splits.X_val, splits.X_cal, splits.X_test])
        assert all_X.shape[0] == 1000

    def test_split_stratification(self):
        """Test stratified splitting preserves class balance."""
        X, y = self.create_test_data(1000, class_balance=0.3)  # 30% malicious
        splits = split_dataset(X, y)

        # Check each split has similar class distribution
        for name, X_split, y_split in [
            ("train", splits.X_train, splits.y_train),
            ("val", splits.X_val, splits.y_val),
            ("cal", splits.X_cal, splits.y_cal),
            ("test", splits.X_test, splits.y_test),
        ]:
            malicious_ratio = np.mean(y_split)
            assert 0.25 <= malicious_ratio <= 0.35, f"{name} split ratio {malicious_ratio:.3f}"

    def test_split_custom_config(self):
        """Test splitting with custom config."""
        X, y = self.create_test_data(1000)
        custom_config = SPLITConfig(
            TRAIN_RATIO=0.7,
            VAL_RATIO=0.1,
            CAL_RATIO=0.1,
            TEST_RATIO=0.1,
            RANDOM_SEED=42,
        )
        splits = split_dataset(X, y, config=custom_config)

        # Allow ±1 due to rounding in train_test_split
        assert 699 <= splits.X_train.shape[0] <= 701
        assert 99 <= splits.X_val.shape[0] <= 101
        assert 99 <= splits.X_cal.shape[0] <= 101
        assert 99 <= splits.X_test.shape[0] <= 101

    def test_split_reproducibility(self):
        """Test that same seed produces same splits."""
        X, y = self.create_test_data(1000)
        splits1 = split_dataset(X, y)
        splits2 = split_dataset(X, y)

        assert np.array_equal(splits1.X_train, splits2.X_train)
        assert np.array_equal(splits1.X_val, splits2.X_val)
        assert np.array_equal(splits1.X_cal, splits2.X_cal)
        assert np.array_equal(splits1.X_test, splits2.X_test)

    def test_split_no_leakage(self):
        """Test no data leakage between splits."""
        X, y = self.create_test_data(2000)
        splits = split_dataset(X, y)

        # Verify no row appears in more than one split
        all_rows = np.vstack([
            splits.X_train, splits.X_val, splits.X_cal, splits.X_test
        ])
        # Convert to tuples for set membership
        row_tuples = [tuple(row) for row in all_rows]
        assert len(set(row_tuples)) == len(row_tuples), "Duplicate rows across splits!"

    def test_split_summary(self):
        """Test summary method returns correct structure."""
        X, y = self.create_test_data(1000)
        splits = split_dataset(X, y)
        summary = splits.summary()

        assert "train" in summary
        assert "validation" in summary
        assert "calibration" in summary
        assert "test" in summary

        for name, info in summary.items():
            assert "samples" in info
            assert "features" in info
            assert "benign_count" in info
            assert "malicious_count" in info


# Fix SPLITConfig reference
class SPLITConfig:
    """Simple config class for testing."""
    def __init__(self, TRAIN_RATIO=0.6, VAL_RATIO=0.15, CAL_RATIO=0.15, TEST_RATIO=0.1, RANDOM_SEED=42, STRATIFY=True):
        self.TRAIN_RATIO = TRAIN_RATIO
        self.VAL_RATIO = VAL_RATIO
        self.CAL_RATIO = CAL_RATIO
        self.TEST_RATIO = TEST_RATIO
        self.RANDOM_SEED = RANDOM_SEED
        self.STRATIFY = STRATIFY


class TestEmberPreprocessor:
    """Tests for EmberPreprocessor."""

    def setup_method(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 2351).astype(np.float32)
        self.X_val = np.random.randn(50, 2351).astype(np.float32)

    def test_fit_transform_default(self):
        """Test fit_transform with default config."""
        preprocessor = EmberPreprocessor()
        X_transformed = preprocessor.fit_transform(self.X_train)

        assert X_transformed.shape == self.X_train.shape
        assert X_transformed.dtype == np.float32
        assert preprocessor.is_fitted is True

        # Check standardization (zero mean, unit variance approximately)
        assert abs(np.mean(X_transformed)) < 0.1
        assert abs(np.std(X_transformed) - 1.0) < 0.1

    def test_fit_then_transform(self):
        """Test separate fit and transform calls."""
        preprocessor = EmberPreprocessor()
        preprocessor.fit(self.X_train)

        X_train_t = preprocessor.transform(self.X_train)
        X_val_t = preprocessor.transform(self.X_val)

        assert X_train_t.shape == self.X_train.shape
        assert X_val_t.shape == self.X_val.shape

        # Validation set should use training statistics
        assert abs(np.mean(X_train_t)) < 0.1

    def test_transform_before_fit_raises(self):
        """Test transform raises error if not fitted."""
        preprocessor = EmberPreprocessor()
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(self.X_train)

    def test_imputation(self):
        """Test NaN imputation works."""
        X = self.X_train.copy()
        X[0, 0] = np.nan
        X[10, 100] = np.nan
        X[20, 500] = np.inf

        preprocessor = EmberPreprocessor()
        X_t = preprocessor.fit_transform(X)

        assert not np.any(np.isnan(X_t))
        assert not np.any(np.isinf(X_t))

    def test_save_load(self, tmp_path):
        """Test saving and loading preprocessor."""
        preprocessor = EmberPreprocessor()
        preprocessor.fit(self.X_train)

        save_path = tmp_path / "test_preprocessor.joblib"
        preprocessor.save(save_path)

        loaded = EmberPreprocessor.load(save_path)
        assert loaded.is_fitted is True

        # Transform should produce same results
        X_t1 = preprocessor.transform(self.X_train[:10])
        X_t2 = loaded.transform(self.X_train[:10])
        assert np.allclose(X_t1, X_t2)

    def test_load_wrong_type(self, tmp_path):
        """Test loading wrong object type fails."""
        # Save something that's not a preprocessor
        import joblib
        joblib.dump({"not": "a preprocessor"}, tmp_path / "wrong.joblib")

        with pytest.raises(TypeError):
            EmberPreprocessor.load(tmp_path / "wrong.joblib")

    def test_scaler_types(self):
        """Test different scaler configurations."""
        X = self.X_train.copy()

        for scaler_type in ["standard", "robust", "minmax", None]:
            config = PREPROCESSINGConfig(SCALER_TYPE=scaler_type)
            preprocessor = EmberPreprocessor(config=config)
            X_t = preprocessor.fit_transform(X)

            assert X_t.shape == X.shape
            assert not np.any(np.isnan(X_t))

    def test_save_unfitted_raises(self, tmp_path):
        """Test saving unfitted preprocessor raises."""
        preprocessor = EmberPreprocessor()
        with pytest.raises(RuntimeError, match="unfitted"):
            preprocessor.save(tmp_path / "unfitted.joblib")


class PREPROCESSINGConfig:
    """Simple config for testing."""
    def __init__(self, IMPUTE_STRATEGY="median", IMPUTE_CONSTANT=0.0,
                 SCALER_TYPE="standard", WITH_CENTERING=True, WITH_SCALING=True,
                 FEATURE_SELECTION=None, SAVE_PREPROCESSOR=True,
                 PREPROCESSOR_FILENAME="preprocessor.joblib", RANDOM_SEED=42):
        self.IMPUTE_STRATEGY = IMPUTE_STRATEGY
        self.IMPUTE_CONSTANT = IMPUTE_CONSTANT
        self.SCALER_TYPE = SCALER_TYPE
        self.WITH_CENTERING = WITH_CENTERING
        self.WITH_SCALING = WITH_SCALING
        self.FEATURE_SELECTION = FEATURE_SELECTION
        self.SAVE_PREPROCESSOR = SAVE_PREPROCESSOR
        self.PREPROCESSOR_FILENAME = PREPROCESSOR_FILENAME
        self.RANDOM_SEED = RANDOM_SEED


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, tmp_path):
        """Test full pipeline: load -> validate -> split -> preprocess."""
        np.random.seed(42)

        # 1. Generate synthetic data (simulating load)
        loader = EmberLoader(random_seed=42)
        X, y = loader.generate_synthetic_data(n_samples=1000)

        # 2. Validate
        from threatlens.dataset.validator import validate_dataset
        report = validate_dataset(X, y)
        assert report.is_valid

        # 3. Split
        splits = split_dataset(X, y)
        assert splits.X_train.shape[0] > 0

        # 4. Preprocess
        preprocessor = EmberPreprocessor()
        preprocessor.fit(splits.X_train)

        splits.X_train = preprocessor.transform(splits.X_train)
        splits.X_val = preprocessor.transform(splits.X_val)
        splits.X_cal = preprocessor.transform(splits.X_cal)
        splits.X_test = preprocessor.transform(splits.X_test)

        # 5. Verify no NaN/Inf in any split
        for name, X_split in [
            ("train", splits.X_train),
            ("val", splits.X_val),
            ("cal", splits.X_cal),
            ("test", splits.X_test),
        ]:
            assert not np.any(np.isnan(X_split)), f"NaN in {name}"
            assert not np.any(np.isinf(X_split)), f"Inf in {name}"

        # 6. Save preprocessor
        preprocessor.save(tmp_path / "preprocessor.joblib")

        print("End-to-end test passed!")


# Import for test
from threatlens.dataset.ember_loader import EmberLoader

if __name__ == "__main__":
    pytest.main([__file__, "-v"])