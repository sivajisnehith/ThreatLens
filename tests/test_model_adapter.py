"""
Unit Tests for ThreatLens Model Adapters

Tests feature vector generation, schema isolation, adapter registry, and
safety checks preventing silent mapping between Custom PE and EMBER 2018 pipelines.
"""

import numpy as np
import pytest

from threatlens.config import EmberFeatureGroups
from threatlens.core.adapter import (
    AdapterValidationError,
    EmberFeatureAdapter,
    FeatureVectorAdapter,
    IncompatibleFeatureError,
    ThreatLensCoreAdapter,
    get_adapter,
    list_adapters,
)
from threatlens.core.extractor import PEFeatureExtractor
from threatlens.core.schema import SCHEMA_VERSION
from tests.test_pe_extractor import create_synthetic_pe


class TestThreatLensCoreAdapter:
    """Tests for ThreatLensCoreAdapter (Custom PE Pipeline)."""

    @pytest.fixture
    def adapter(self):
        return ThreatLensCoreAdapter()

    @pytest.fixture
    def sample_features(self):
        pe_bytes = create_synthetic_pe(num_sections=2)
        extractor = PEFeatureExtractor(data=pe_bytes)
        return extractor.extract()

    def test_adapter_properties(self, adapter):
        """Verify pipeline identification, versioning, and dimensions."""
        assert adapter.pipeline_name == "threatlens_custom_pe"
        assert adapter.schema_version == SCHEMA_VERSION
        assert adapter.feature_dim > 50
        assert len(adapter.feature_names) == adapter.feature_dim
        assert "file_size" in adapter.feature_names
        assert "mean_section_entropy" in adapter.feature_names
        assert "char_executable_image" in adapter.feature_names

    def test_transform_valid_features(self, adapter, sample_features):
        """Verify deterministic conversion into 1D float32 vector."""
        vec = adapter.transform(sample_features)
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == adapter.feature_dim
        assert vec.dtype == np.float32
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

        # Check specific mapped values
        file_size_idx = adapter.feature_names.index("file_size")
        assert vec[file_size_idx] == float(sample_features["file_size"])

    def test_transform_batch(self, adapter, sample_features):
        """Verify batch transformation creates correct 2D matrix."""
        batch = [sample_features, sample_features, sample_features]
        mat = adapter.transform_batch(batch)
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (3, adapter.feature_dim)
        assert mat.dtype == np.float32

        empty_mat = adapter.transform_batch([])
        assert empty_mat.shape == (0, adapter.feature_dim)

    def test_rejects_incompatible_schema_version(self, adapter, sample_features):
        """Verify adapter rejects feature dictionaries with incompatible version."""
        corrupted = dict(sample_features)
        corrupted["schema_version"] = "2.0.0"
        with pytest.raises(IncompatibleFeatureError):
            adapter.transform(corrupted)

    def test_rejects_invalid_type(self, adapter):
        """Verify non-dictionary input raises AdapterValidationError."""
        with pytest.raises(AdapterValidationError):
            adapter.transform("not_a_dictionary")  # type: ignore


class TestEmberFeatureAdapter:
    """Tests for EmberFeatureAdapter (EMBER 2018 Pipeline)."""

    @pytest.fixture
    def adapter(self):
        return EmberFeatureAdapter()

    @pytest.fixture
    def custom_pe_features(self):
        pe_bytes = create_synthetic_pe(num_sections=1)
        extractor = PEFeatureExtractor(data=pe_bytes)
        return extractor.extract()

    def test_adapter_properties(self, adapter):
        """Verify EMBER 2018 feature specification and 2,351 dimensions."""
        assert adapter.pipeline_name == "ember_2018"
        assert adapter.feature_dim == 2351
        assert adapter.feature_dim == EmberFeatureGroups.TOTAL_FEATURES
        assert len(adapter.feature_names) == 2351

    def test_strictly_rejects_custom_pe_features(self, adapter, custom_pe_features):
        """
        CRITICAL SAFETY TEST:
        Verify that EmberFeatureAdapter strictly rejects Custom PE features
        and refuses to silently map or pad them into EMBER 2018 vectors.
        """
        with pytest.raises(IncompatibleFeatureError) as exc_info:
            adapter.transform(custom_pe_features)
        assert "Incompatible feature schema" in str(exc_info.value)
        assert "strictly prohibited" in str(exc_info.value)

        with pytest.raises(IncompatibleFeatureError):
            adapter.validate_input(custom_pe_features)

    def test_accepts_valid_ember_vector(self, adapter):
        """Verify valid 2,351-dimensional array passes transformation."""
        valid_vec = np.random.randn(2351).astype(np.float32)
        out = adapter.transform(valid_vec)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2351,)
        assert out.dtype == np.float32

    def test_rejects_wrong_dimension_array(self, adapter):
        """Verify wrong-sized arrays raise AdapterValidationError."""
        wrong_size_vec = np.zeros(100, dtype=np.float32)
        with pytest.raises(AdapterValidationError) as exc_info:
            adapter.transform(wrong_size_vec)
        assert "must have exactly 2351 dimensions" in str(exc_info.value)


class TestAdapterRegistry:
    """Tests for adapter discovery and instantiation."""

    def test_list_adapters(self):
        """Verify both pipelines are present in registry."""
        adapters = list_adapters()
        assert "threatlens_core" in adapters
        assert "ember_2018" in adapters

    def test_get_adapter_instances(self):
        """Verify retrieval by name returns correct adapter instances."""
        core_adapter = get_adapter("threatlens_core")
        assert isinstance(core_adapter, ThreatLensCoreAdapter)

        ember_adapter = get_adapter("ember_2018")
        assert isinstance(ember_adapter, EmberFeatureAdapter)

    def test_get_unknown_adapter_raises(self):
        """Verify requesting unknown adapter name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_adapter("non_existent_adapter")
        assert "Unknown adapter" in str(exc_info.value)
