"""
Unit Tests for EMBER Loader Module
"""

import numpy as np
import pytest

from threatlens.config import DATA_CONFIG, EmberFeatureGroups
from threatlens.dataset.ember_loader import EmberLoader, load_ember_data


class TestEmberLoader:
    """Tests for EmberLoader class."""

    def setup_method(self):
        """Create test fixtures."""
        self.loader = EmberLoader(random_seed=42)

    def test_generate_synthetic_data_default(self):
        """Test synthetic data generation with defaults."""
        X, y = self.loader.generate_synthetic_data()

        assert X.shape == (1000, 2351)
        assert y.shape == (1000,)
        assert X.dtype == np.float32
        assert y.dtype == np.int32
        assert set(np.unique(y)).issubset({0, 1})

    def test_generate_synthetic_data_custom(self):
        """Test synthetic data generation with custom params."""
        X, y = self.loader.generate_synthetic_data(n_samples=500, class_balance=0.3)

        assert X.shape == (500, 2351)
        assert y.shape == (500,)
        # ~30% malicious (label=1)
        malicious_count = np.sum(y == 1)
        assert 120 <= malicious_count <= 180  # Allow some variance

    def test_generate_synthetic_feature_groups(self):
        """Test synthetic data has non-zero values in all feature groups."""
        X, y = self.loader.generate_synthetic_data(n_samples=100)

        for group_name in EmberFeatureGroups.get_all_group_names():
            start, end = EmberFeatureGroups.get_group_indices(group_name)
            group_data = X[:, start:end]
            # Check not all zeros (some features should be non-zero)
            assert not np.all(group_data == 0), f"Group {group_name} is all zeros"

    def test_load_or_generate_train(self):
        """Test load_or_generate for train split."""
        X, y = self.loader.load_or_generate(split="train", max_samples=100)

        assert X.shape == (100, 2351)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_load_or_generate_test(self):
        """Test load_or_generate for test split."""
        X, y = self.loader.load_or_generate(split="test", max_samples=50)

        assert X.shape == (50, 2351)
        assert y.shape == (50,)

    def test_load_or_generate_invalid_split(self):
        """Test load_or_generate raises on invalid split."""
        with pytest.raises(ValueError, match="Unknown split"):
            self.loader.load_or_generate(split="invalid")

    def test_loader_with_custom_data_dir(self, tmp_path):
        """Test loader works with custom data directory."""
        custom_dir = tmp_path / "custom_data"
        loader = EmberLoader(data_dir=custom_dir, random_seed=123)

        # Should generate synthetic since no EMBER data
        X, y = loader.load_or_generate(split="train", max_samples=50)
        assert X.shape == (50, 2351)


class TestConvenienceFunction:
    """Tests for load_ember_data convenience function."""

    def test_load_ember_data_train(self):
        X, y = load_ember_data(split="train", max_samples=50, generate_if_missing=True)
        assert X.shape == (50, 2351)

    def test_load_ember_data_test(self):
        X, y = load_ember_data(split="test", max_samples=20, generate_if_missing=True)
        assert X.shape == (20, 2351)

    def test_reproducibility(self):
        """Test that same seed produces same synthetic data."""
        X1, y1 = load_ember_data(split="train", max_samples=100, random_seed=42)
        X2, y2 = load_ember_data(split="train", max_samples=100, random_seed=42)

        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        X1, _ = load_ember_data(split="train", max_samples=100, random_seed=42)
        X2, _ = load_ember_data(split="train", max_samples=100, random_seed=123)

        assert not np.array_equal(X1, X2)


class TestFeatureExtraction:
    """Test internal feature extraction methods."""

    def test_extract_histogram(self):
        """Test histogram extraction."""
        loader = EmberLoader(random_seed=42)
        # Valid histogram
        hist = loader._extract_histogram({"histogram": [1.0] * 256})
        assert hist.shape == (256,)
        assert np.allclose(hist, 1.0)

        # Missing histogram -> zeros
        hist = loader._extract_histogram({})
        assert hist.shape == (256,)
        assert np.all(hist == 0)

        # Wrong length -> zeros
        hist = loader._extract_histogram({"histogram": [1.0] * 100})
        assert hist.shape == (256,)
        assert np.all(hist == 0)

    def test_extract_general(self):
        """Test general feature extraction."""
        loader = EmberLoader(random_seed=42)
        general = loader._extract_general({
            "general": {
                "size": 1000,
                "vsize": 2000,
                "has_debug": 1,
                "exports": 10,
                "imports": 50,
                "has_relocations": 1,
                "has_resources": 1,
                "has_signature": 0,
                "has_tls": 1,
                "symbols": 5,
            }
        })
        assert general.shape == (10,)
        assert general[0] == 1000  # size

    def test_extract_header(self):
        """Test header feature extraction."""
        loader = EmberLoader(random_seed=42)
        header = loader._extract_header({
            "header": {
                "coff": {"timestamp": 12345, "machine": 332},
                "optional": {"magic": 523, "size_of_code": 4096}
            }
        })
        assert header.shape == (61,)
        assert header[0] == 12345  # timestamp (COFF index 0)
        assert header[1] == 332    # machine (COFF index 1)
        # magic is at index 20 (7 COFF + 13 padding)
        assert header[20] == 523   # magic (optional header index 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])