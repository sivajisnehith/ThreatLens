"""
Unit Tests for ThreatLens PE Feature Schema and Validation

Tests schema specification, type validation, range constraints, strict mode,
feature ordering, and schema version consistency.
"""

import sys
import pytest

from threatlens.core.extractor import PEFeatureExtractor
from threatlens.core.schema import (
    FEATURE_ORDER,
    PE_FEATURE_SCHEMA,
    SCHEMA_VERSION,
    SchemaValidationError,
    get_feature_order,
    validate_feature_schema,
    validate_features,
)
from tests.test_pe_extractor import create_synthetic_pe


class TestFeatureSchema:
    """Tests for schema definition and integrity."""

    def test_schema_version_defined(self):
        """Verify schema version is defined and semantic."""
        assert SCHEMA_VERSION == "1.0.0"

    def test_feature_order_matches_schema(self):
        """Verify get_feature_order() matches canonical schema keys."""
        order = get_feature_order()
        assert order == list(PE_FEATURE_SCHEMA.keys())
        assert len(order) == len(PE_FEATURE_SCHEMA)
        assert "schema_version" == order[0]
        assert "file_size" in order
        assert "sections" in order
        assert "imports" in order

    def test_all_schema_entries_have_descriptions_and_types(self):
        """Verify every schema property has type and description."""
        for name, spec in PE_FEATURE_SCHEMA.items():
            assert "type" in spec, f"Schema property {name} missing type"
            assert "description" in spec, f"Schema property {name} missing description"
            assert len(spec["description"]) > 0


class TestSchemaValidation:
    """Tests for validate_feature_schema and validate_features."""

    @pytest.fixture
    def valid_features(self):
        """Fixture returning valid features from synthetic PE."""
        pe_bytes = create_synthetic_pe(num_sections=2)
        extractor = PEFeatureExtractor(data=pe_bytes)
        return extractor.extract()

    def test_valid_synthetic_features_pass_strict_validation(self, valid_features):
        """Verify extractor output passes strict validation without errors."""
        is_valid, errors = validate_feature_schema(valid_features, strict=True)
        assert is_valid is True, f"Validation failed with errors: {errors}"
        assert len(errors) == 0
        assert validate_features(valid_features, strict=True) is True

    def test_valid_benign_system_features_pass_strict_validation(self):
        """Verify extractor output from python.exe passes strict validation."""
        extractor = PEFeatureExtractor(sys.executable)
        features = extractor.extract()
        is_valid, errors = validate_feature_schema(features, strict=True)
        assert is_valid is True, f"System binary validation failed: {errors}"
        assert len(errors) == 0

    def test_schema_version_mismatch_fails(self, valid_features):
        """Verify that incorrect schema version fails validation."""
        corrupted = dict(valid_features)
        corrupted["schema_version"] = "9.9.9"
        is_valid, errors = validate_feature_schema(corrupted, strict=False)
        assert is_valid is False
        assert any("version mismatch" in e.lower() for e in errors)

    def test_missing_required_feature_fails(self, valid_features):
        """Verify that omitting a required feature fails validation."""
        corrupted = dict(valid_features)
        del corrupted["file_size"]
        is_valid, errors = validate_feature_schema(corrupted, strict=False)
        assert is_valid is False
        assert any("file_size" in e for e in errors)

    def test_strict_mode_rejects_unexpected_keys(self, valid_features):
        """Verify that strict mode rejects unrecognized keys."""
        corrupted = dict(valid_features)
        corrupted["unrecognized_rogue_key"] = 12345
        is_valid, errors = validate_feature_schema(corrupted, strict=True)
        assert is_valid is False
        assert any("unrecognized_rogue_key" in e for e in errors)

        # In non-strict mode, extra keys do not fail validation
        is_valid_non_strict, _ = validate_feature_schema(corrupted, strict=False)
        assert is_valid_non_strict is True

    def test_type_mismatch_fails(self, valid_features):
        """Verify that passing incorrect type fails validation."""
        corrupted = dict(valid_features)
        corrupted["file_size"] = "not_an_integer"
        is_valid, errors = validate_feature_schema(corrupted, strict=False)
        assert is_valid is False
        assert any("type mismatch" in e.lower() for e in errors)

    def test_out_of_bounds_numerical_constraint(self, valid_features):
        """Verify min/max bounds checking."""
        corrupted = dict(valid_features)
        corrupted["file_size"] = -500  # min is 0
        is_valid, errors = validate_feature_schema(corrupted, strict=False)
        assert is_valid is False
        assert any("less than min" in e.lower() for e in errors)

    def test_raise_on_error_convenience(self, valid_features):
        """Verify validate_features raises SchemaValidationError when raise_on_error=True."""
        corrupted = dict(valid_features)
        corrupted["schema_version"] = "invalid"
        with pytest.raises(SchemaValidationError):
            validate_features(corrupted, strict=True, raise_on_error=True)

    def test_safe_handling_of_optional_characteristics(self, valid_features):
        """Verify 64-bit PE null BaseOfData is accepted as valid."""
        features = dict(valid_features)
        features["optional_header_base_of_data"] = None
        is_valid, errors = validate_feature_schema(features, strict=True)
        assert is_valid is True, f"BaseOfData=None should be valid for PE32+, got errors: {errors}"
