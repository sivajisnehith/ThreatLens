"""
ThreatLens Core Module

Independent PE static feature extraction engine, schema validation,
and model adapter interfaces.
"""

from threatlens.core.adapter import (
    AdapterValidationError,
    EmberFeatureAdapter,
    FeatureVectorAdapter,
    IncompatibleFeatureError,
    ThreatLensCoreAdapter,
    get_adapter,
    list_adapters,
)
from threatlens.core.extractor import (
    PEFeatureExtractor,
    PEParseError,
    calculate_entropy,
    extract_pe_features,
)
from threatlens.core.schema import (
    FEATURE_ORDER,
    PE_FEATURE_SCHEMA,
    SCHEMA_VERSION,
    FeatureSpec,
    SchemaValidationError,
    get_feature_order,
    validate_feature_schema,
    validate_features,
)

__all__ = [
    # Extractor
    "PEFeatureExtractor",
    "PEParseError",
    "calculate_entropy",
    "extract_pe_features",
    # Schema
    "SCHEMA_VERSION",
    "PE_FEATURE_SCHEMA",
    "FEATURE_ORDER",
    "FeatureSpec",
    "SchemaValidationError",
    "get_feature_order",
    "validate_feature_schema",
    "validate_features",
    # Adapters
    "FeatureVectorAdapter",
    "ThreatLensCoreAdapter",
    "EmberFeatureAdapter",
    "IncompatibleFeatureError",
    "AdapterValidationError",
    "get_adapter",
    "list_adapters",
]
