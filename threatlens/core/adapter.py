"""
Model Adapter Interface for ThreatLens Feature Pipelines

Provides a strict abstraction layer between extracted static PE features and
downstream machine learning models. Enforces strict schema verification, feature
ordering, and explicit isolation between the EMBER 2018 benchmark pipeline
and the ThreatLens Custom PE pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np

from threatlens.config import EmberFeatureGroups
from threatlens.core.schema import SCHEMA_VERSION, validate_features
from threatlens.utils.logger import get_logger

logger = get_logger("adapter")


class IncompatibleFeatureError(Exception):
    """Raised when an adapter receives feature representations from an incompatible schema."""
    pass


class AdapterValidationError(Exception):
    """Raised when feature vectors fail dimension, naming, or range validation."""
    pass


class FeatureVectorAdapter(ABC):
    """
    Abstract base class for converting PE feature sets into fixed-size,
    deterministic numerical vectors for model training and inference.
    """

    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Return the pipeline identifier (e.g. 'threatlens_custom_pe', 'ember_2018')."""
        pass

    @property
    @abstractmethod
    def schema_version(self) -> str:
        """Return the expected schema version."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the exact dimensionality of the output feature vector."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return the ordered list of feature names for the output vector."""
        pass

    @abstractmethod
    def validate_input(self, features: Any) -> bool:
        """
        Validate input features against expected schema and constraints.

        Args:
            features: Input feature dictionary or array.

        Returns:
            True if compatible, raises IncompatibleFeatureError or AdapterValidationError otherwise.
        """
        pass

    @abstractmethod
    def transform(self, extracted_features: Dict[str, Any]) -> np.ndarray:
        """
        Convert extracted feature dictionary into a verified numerical numpy array.

        Args:
            extracted_features: Dictionary of extracted features.

        Returns:
            NumPy array of shape (feature_dim,) with dtype np.float32.
        """
        pass

    def transform_batch(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform a list of feature dictionaries into a 2D feature matrix.

        Args:
            features_list: List of feature dictionaries.

        Returns:
            NumPy array of shape (len(features_list), feature_dim).
        """
        if not features_list:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.vstack([self.transform(f) for f in features_list])


class ThreatLensCoreAdapter(FeatureVectorAdapter):
    """
    Adapter for the ThreatLens Custom PE Feature Schema (pefile-based).
    Produces a verified, deterministic, fixed-size numerical vector.
    """

    def __init__(self):
        self._schema_version: str = SCHEMA_VERSION
        self._feature_spec = self._build_feature_spec()

    @property
    def pipeline_name(self) -> str:
        return "threatlens_custom_pe"

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def feature_dim(self) -> int:
        return len(self._feature_spec)

    @property
    def feature_names(self) -> List[str]:
        return [s["name"] for s in self._feature_spec]

    def _build_feature_spec(self) -> List[Dict[str, Any]]:
        """Construct the deterministic specification mapping features to vector indices."""
        spec: List[Dict[str, Any]] = []

        # 1. Scalar Header & File Information
        scalars = [
            ("file_size", lambda f: f.get("file_size", 0)),
            ("machine_architecture", lambda f: f.get("machine_architecture", 0)),
            ("pe_timestamp", lambda f: f.get("pe_timestamp", 0)),
            ("checksum", lambda f: f.get("checksum", 0)),
            ("entry_point", lambda f: f.get("entry_point", 0)),
            ("image_base", lambda f: f.get("image_base", 0)),
            ("subsystem", lambda f: f.get("subsystem", 0)),
            ("dos_header_magic", lambda f: f.get("dos_header_magic", 0)),
            ("dos_header_pe_link", lambda f: f.get("dos_header_pe_link", 0)),
            ("file_header_characteristics", lambda f: f.get("file_header_characteristics", 0)),
            ("file_header_number_of_sections", lambda f: f.get("file_header_number_of_sections", 0)),
            ("file_header_pointer_symbol_table", lambda f: f.get("file_header_pointer_symbol_table", 0)),
            ("file_header_number_of_symbols", lambda f: f.get("file_header_number_of_symbols", 0)),
            ("file_header_size_optional_header", lambda f: f.get("file_header_size_optional_header", 0)),
            ("optional_header_magic", lambda f: f.get("optional_header_magic", 0)),
            ("optional_header_major_linker_version", lambda f: f.get("optional_header_major_linker_version", 0)),
            ("optional_header_minor_linker_version", lambda f: f.get("optional_header_minor_linker_version", 0)),
            ("optional_header_size_of_code", lambda f: f.get("optional_header_size_of_code", 0)),
            ("optional_header_size_of_initialized_data", lambda f: f.get("optional_header_size_of_initialized_data", 0)),
            ("optional_header_size_of_uninitialized_data", lambda f: f.get("optional_header_size_of_uninitialized_data", 0)),
            ("optional_header_base_of_code", lambda f: f.get("optional_header_base_of_code", 0)),
            ("optional_header_base_of_data", lambda f: f.get("optional_header_base_of_data") or 0),
            ("optional_header_image_size", lambda f: f.get("optional_header_image_size", 0)),
            ("optional_header_size_of_headers", lambda f: f.get("optional_header_size_of_headers", 0)),
            ("optional_header_checksum", lambda f: f.get("optional_header_checksum", 0)),
            ("optional_header_dll_characteristics", lambda f: f.get("optional_header_dll_characteristics", 0)),
            ("optional_header_size_of_stack_reserve", lambda f: f.get("optional_header_size_of_stack_reserve", 0)),
            ("optional_header_size_of_stack_commit", lambda f: f.get("optional_header_size_of_stack_commit", 0)),
            ("optional_header_size_of_heap_reserve", lambda f: f.get("optional_header_size_of_heap_reserve", 0)),
            ("optional_header_size_of_heap_commit", lambda f: f.get("optional_header_size_of_heap_commit", 0)),
            ("optional_header_number_of_rva_and_sizes", lambda f: f.get("optional_header_number_of_rva_and_sizes", 0)),
            ("optional_header_section_alignment", lambda f: f.get("optional_header_section_alignment", 0)),
            ("optional_header_file_alignment", lambda f: f.get("optional_header_file_alignment", 0)),
        ]
        for name, fn in scalars:
            spec.append({"name": name, "extractor": fn})

        # 2. File Header Characteristic Flags
        file_flags = [
            "relocs_stripped", "executable_image", "line_nums_stripped", "local_syms_stripped",
            "large_address_aware", "32bit_machine", "debug_stripped", "removable_run_from_swap",
            "net_run_from_swap", "system", "dll", "up_system_only",
        ]
        for flag in file_flags:
            spec.append({
                "name": f"char_{flag}",
                "extractor": lambda f, fl=flag: 1.0 if f.get("characteristics", {}).get(fl, False) else 0.0,
            })

        # 3. DLL Characteristics Flags
        dll_flags = [
            "high_entropy_va", "dynamic_base", "force_integrity", "nx_compat",
            "no_isolation", "no_seh", "no_bind", "appcontainer",
            "wdm_driver", "guard_cf", "terminal_server_aware",
        ]
        for flag in dll_flags:
            spec.append({
                "name": f"dll_{flag}",
                "extractor": lambda f, fl=flag: 1.0 if f.get("dll_characteristics_flags", {}).get(fl, False) else 0.0,
            })

        # 4. Section Aggregates & Statistics
        def _get_entropies(f: Dict[str, Any]) -> List[float]:
            sections = f.get("sections", [])
            return [s.get("entropy", 0.0) for s in sections] if sections else [0.0]

        section_stats = [
            ("section_count", lambda f: float(len(f.get("sections", [])))),
            ("total_section_raw_size", lambda f: float(sum(s.get("size_of_raw_data", 0) for s in f.get("sections", [])))),
            ("total_section_virtual_size", lambda f: float(sum(s.get("virtual_size", 0) for s in f.get("sections", [])))),
            ("mean_section_entropy", lambda f: float(np.mean(_get_entropies(f)))),
            ("min_section_entropy", lambda f: float(np.min(_get_entropies(f)))),
            ("max_section_entropy", lambda f: float(np.max(_get_entropies(f)))),
            ("executable_section_count", lambda f: float(sum(1 for s in f.get("sections", []) if s.get("permissions", {}).get("executable", False)))),
            ("writable_section_count", lambda f: float(sum(1 for s in f.get("sections", []) if s.get("permissions", {}).get("writable", False)))),
            ("readable_section_count", lambda f: float(sum(1 for s in f.get("sections", []) if s.get("permissions", {}).get("readable", False)))),
            ("code_section_count", lambda f: float(sum(1 for s in f.get("sections", []) if s.get("permissions", {}).get("contains_code", False)))),
            ("data_section_count", lambda f: float(sum(1 for s in f.get("sections", []) if s.get("permissions", {}).get("contains_initialized_data", False)))),
            ("shared_section_count", lambda f: float(sum(1 for s in f.get("sections", []) if s.get("permissions", {}).get("shared", False)))),
        ]
        for name, fn in section_stats:
            spec.append({"name": name, "extractor": fn})

        # 5. Imports & Exports Aggregates
        import_export_stats = [
            ("number_of_imported_dlls", lambda f: float(f.get("number_of_imported_dlls", 0))),
            ("number_of_imported_functions", lambda f: float(f.get("number_of_imported_functions", 0))),
            ("has_empty_dll_names", lambda f: 1.0 if f.get("unusual_import_characteristics", {}).get("has_empty_dll_names", False) else 0.0),
            ("has_ordinal_only_imports", lambda f: 1.0 if f.get("unusual_import_characteristics", {}).get("has_ordinal_only_imports", False) else 0.0),
            ("suspicious_import_count", lambda f: float(f.get("unusual_import_characteristics", {}).get("suspicious_import_count", 0))),
            ("number_of_exports", lambda f: float(f.get("number_of_exports", 0))),
        ]
        for name, fn in import_export_stats:
            spec.append({"name": name, "extractor": fn})

        # 6. Resources, Overlay, Security & Debug
        misc_stats = [
            ("has_resources", lambda f: 1.0 if f.get("has_resources", False) else 0.0),
            ("resource_count", lambda f: float(f.get("resource_count", 0))),
            ("resource_type_count", lambda f: float(len(f.get("resource_types", [])))),
            ("total_resource_size", lambda f: float(sum(r.get("size", 0) for r in f.get("resources", [])))),
            ("has_overlay", lambda f: 1.0 if f.get("has_overlay", False) else 0.0),
            ("overlay_size", lambda f: float(f.get("overlay_size", 0))),
            ("overlay_entropy", lambda f: float(f.get("overlay_entropy", 0.0))),
            ("certificate_present", lambda f: 1.0 if f.get("certificate_table", {}).get("present", False) else 0.0),
            ("certificate_size", lambda f: float(f.get("certificate_table", {}).get("size", 0))),
            ("debug_entries_count", lambda f: float(f.get("debug_entries_count", 0))),
            ("has_debug_data", lambda f: 1.0 if f.get("has_debug_data", False) else 0.0),
        ]
        for name, fn in misc_stats:
            spec.append({"name": name, "extractor": fn})

        return spec

    def validate_input(self, features: Any) -> bool:
        """Validate that input dictionary complies with the ThreatLens custom schema."""
        if not isinstance(features, dict):
            raise AdapterValidationError(f"Expected feature dictionary, got {type(features).__name__}")

        version = features.get("schema_version")
        if version != self.schema_version:
            raise IncompatibleFeatureError(
                f"Schema version mismatch: adapter requires '{self.schema_version}', "
                f"input has '{version}'"
            )

        if not validate_features(features, strict=False):
            raise AdapterValidationError("Input features failed schema validation constraints")

        return True

    def transform(self, extracted_features: Dict[str, Any]) -> np.ndarray:
        """Transform extracted features dictionary into a validated float32 vector."""
        self.validate_input(extracted_features)

        vector = np.zeros(self.feature_dim, dtype=np.float32)
        for idx, spec in enumerate(self._feature_spec):
            try:
                val = spec["extractor"](extracted_features)
                if val is None:
                    vector[idx] = 0.0
                elif isinstance(val, bool):
                    vector[idx] = 1.0 if val else 0.0
                else:
                    vector[idx] = float(val)
            except Exception as e:
                logger.warning(f"Error extracting adapter feature '{spec['name']}': {e}. Using 0.0.")
                vector[idx] = 0.0

        return vector


class EmberFeatureAdapter(FeatureVectorAdapter):
    """
    Adapter for the official EMBER 2018 benchmark feature representation.
    Guarantees strict 2,351-dimensional alignment and explicitly rejects
    incompatible custom PE feature dictionaries.
    """

    def __init__(self):
        self._target_dim: int = EmberFeatureGroups.TOTAL_FEATURES  # 2351
        self._schema_version: str = "ember_2018_v2"
        self._feature_names: List[str] = self._build_ember_feature_names()

    @property
    def pipeline_name(self) -> str:
        return "ember_2018"

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def feature_dim(self) -> int:
        return self._target_dim

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def _build_ember_feature_names(self) -> List[str]:
        """Generate canonical names for the 2,351 EMBER features across the 8 groups."""
        names = []
        for group in EmberFeatureGroups.get_all_group_names():
            start, end = EmberFeatureGroups.get_group_indices(group)
            for i in range(start, end):
                names.append(f"ember_{group}_{i}")
        return names

    def validate_input(self, features: Any) -> bool:
        """
        Validate EMBER feature input.
        Strictly rejects ThreatLens custom PE feature dictionaries.
        """
        # Reject custom PE feature dictionaries
        if isinstance(features, dict):
            # Check for signatures of custom PE extraction
            if "dos_header" in features or "schema_version" in features or "sections" in features:
                raise IncompatibleFeatureError(
                    "Incompatible feature schema: EmberFeatureAdapter strictly requires official "
                    "EMBER 2018 feature representations (2,351 dimensions). "
                    "Received ThreatLens custom PE feature dictionary. "
                    "Silent mapping or zero-padding custom features into EMBER feature vectors is "
                    "strictly prohibited."
                )

        if isinstance(features, np.ndarray):
            if features.ndim == 1 and len(features) != self.feature_dim:
                raise AdapterValidationError(
                    f"EMBER feature vector must have exactly {self.feature_dim} dimensions, got {len(features)}"
                )
            if features.ndim == 2 and features.shape[1] != self.feature_dim:
                raise AdapterValidationError(
                    f"EMBER feature matrix must have {self.feature_dim} columns, got {features.shape[1]}"
                )
            return True

        raise IncompatibleFeatureError(
            f"EmberFeatureAdapter requires a 2,351-dimensional numpy array or EMBER record, "
            f"got {type(features).__name__}"
        )

    def transform(self, extracted_features: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
        """
        Verify and return EMBER feature vector.
        Strictly rejects incompatible custom PE feature representations.
        """
        self.validate_input(extracted_features)

        if isinstance(extracted_features, np.ndarray):
            return extracted_features.astype(np.float32).ravel()

        raise IncompatibleFeatureError(
            "Cannot transform non-array representation into EMBER feature vector without "
            "official EMBER 2018 raw byte extraction."
        )


# Adapter registry
ADAPTER_REGISTRY: Dict[str, Any] = {
    "threatlens_core": ThreatLensCoreAdapter,
    "ember_2018": EmberFeatureAdapter,
}


def get_adapter(adapter_name: str) -> FeatureVectorAdapter:
    """Retrieve an instantiated adapter by name."""
    if adapter_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter '{adapter_name}'. Available adapters: {list_adapters()}")
    return ADAPTER_REGISTRY[adapter_name]()


def list_adapters() -> List[str]:
    """List all registered adapter names."""
    return list(ADAPTER_REGISTRY.keys())


__all__ = [
    "FeatureVectorAdapter",
    "ThreatLensCoreAdapter",
    "EmberFeatureAdapter",
    "IncompatibleFeatureError",
    "AdapterValidationError",
    "get_adapter",
    "list_adapters",
]
