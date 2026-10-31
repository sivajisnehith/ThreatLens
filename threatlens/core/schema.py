"""
ThreatLens Core Feature Schema

Defines the schema for static PE file features extracted by the PEFeatureExtractor.
This includes feature names, descriptions, data types, constraints, missing-value policies,
canonical feature ordering, and strict validation routines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from threatlens.utils.logger import get_logger

logger = get_logger("core_schema")

# Schema version identifier
SCHEMA_VERSION: str = "1.0.0"


class SchemaValidationError(Exception):
    """Raised when extracted features fail schema validation."""
    pass


@dataclass(frozen=True)
class FeatureSpec:
    """Specification and constraints for a single feature."""
    name: str
    data_type: Union[type, Tuple[type, ...], str]
    description: str
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    allowed_values: Optional[Union[List[Any], Dict[Any, str]]] = None
    missing_policy: str = "error"  # 'error', 'nullable', 'default'
    default_value: Any = None
    is_optional: bool = False


# Canonical specification for all top-level features
PE_FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    # Schema metadata
    "schema_version": {
        "type": "string",
        "description": "Version of the ThreatLens PE feature schema.",
        "missing_policy": "error",
    },

    # File information
    "file_size": {
        "type": "integer",
        "description": "Total size of the executable file in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "file_type": {
        "type": "string",
        "description": "Type of PE binary (e.g. PE32, PE32+).",
        "values": ["PE32", "PE32+", "UNKNOWN"],
        "missing_policy": "error",
    },
    "machine_architecture": {
        "type": "integer",
        "description": "Target processor machine code from COFF header (e.g. 0x14C for i386, 0x8664 for AMD64).",
        "min": 0,
        "missing_policy": "error",
    },
    "machine_name": {
        "type": "string",
        "description": "Human-readable processor architecture name.",
        "missing_policy": "error",
    },
    "pe_timestamp": {
        "type": "integer",
        "description": "Compilation/creation Unix timestamp from COFF header.",
        "min": 0,
        "missing_policy": "error",
    },
    "pe_timestamp_iso": {
        "type": "string",
        "description": "ISO 8601 formatted compilation timestamp string.",
        "missing_policy": "default",
        "default": "",
    },
    "checksum": {
        "type": "integer",
        "description": "Optional header checksum value.",
        "min": 0,
        "missing_policy": "error",
    },

    # General PE Header Fields
    "entry_point": {
        "type": "integer",
        "description": "RVA of the entry point of the executable.",
        "min": 0,
        "missing_policy": "error",
    },
    "image_base": {
        "type": "integer",
        "description": "Preferred base address for loading the executable into memory.",
        "min": 0,
        "missing_policy": "error",
    },
    "subsystem": {
        "type": "integer",
        "description": "The operating system subsystem required to run this image.",
        "min": 0,
        "missing_policy": "error",
    },
    "subsystem_name": {
        "type": "string",
        "description": "Human-readable name of the operating system subsystem.",
        "missing_policy": "default",
        "default": "UNKNOWN",
    },

    # DOS Header Characteristics
    "dos_header_magic": {
        "type": "integer",
        "description": "Magic number in DOS header (0x5A4D for 'MZ').",
        "min": 0,
        "missing_policy": "error",
    },
    "dos_header_pe_link": {
        "type": "integer",
        "description": "File offset to the PE (NT) header (e_lfanew).",
        "min": 0,
        "missing_policy": "error",
    },
    "dos_header": {
        "type": "dict",
        "description": "Complete decoded DOS header structure.",
        "missing_policy": "error",
    },

    # COFF File Header Characteristics
    "file_header_characteristics": {
        "type": "integer",
        "description": "Raw 16-bit characteristics bitmask defined in the COFF file header.",
        "min": 0,
        "missing_policy": "error",
    },
    "file_header_number_of_sections": {
        "type": "integer",
        "description": "Number of sections specified in the COFF file header.",
        "min": 0,
        "missing_policy": "error",
    },
    "file_header_pointer_symbol_table": {
        "type": "integer",
        "description": "File pointer to COFF symbol table, or zero if none.",
        "min": 0,
        "missing_policy": "error",
    },
    "file_header_number_of_symbols": {
        "type": "integer",
        "description": "Number of entries in COFF symbol table.",
        "min": 0,
        "missing_policy": "error",
    },
    "file_header_size_optional_header": {
        "type": "integer",
        "description": "Size of the optional header in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "characteristics": {
        "type": "dict",
        "description": "Detailed boolean breakdown of COFF file header characteristics.",
        "missing_policy": "error",
    },

    # Optional Header Characteristics
    "optional_header_magic": {
        "type": "integer",
        "description": "Magic number indicating PE32 (0x10B) or PE32+ (0x20B).",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_major_linker_version": {
        "type": "integer",
        "description": "Major version number of the linker.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_minor_linker_version": {
        "type": "integer",
        "description": "Minor version number of the linker.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_code": {
        "type": "integer",
        "description": "Size of the code (.text) section in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_initialized_data": {
        "type": "integer",
        "description": "Size of the initialized data section in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_uninitialized_data": {
        "type": "integer",
        "description": "Size of the uninitialized data section in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_base_of_code": {
        "type": "integer",
        "description": "RVA of the beginning of the code section relative to ImageBase.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_base_of_data": {
        "type": ["integer", "null"],
        "description": "RVA of the beginning of the data section (PE32 only; null for PE32+).",
        "missing_policy": "nullable",
    },
    "optional_header_image_size": {
        "type": "integer",
        "description": "Total virtual memory size of the image including headers.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_headers": {
        "type": "integer",
        "description": "Combined size of MS-DOS stub, PE header, and section headers.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_checksum": {
        "type": "integer",
        "description": "Image file checksum from Optional Header.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_dll_characteristics": {
        "type": "integer",
        "description": "DLL characteristics bitmask from Optional Header.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_stack_reserve": {
        "type": "integer",
        "description": "Size of stack to reserve in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_stack_commit": {
        "type": "integer",
        "description": "Size of stack to commit in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_heap_reserve": {
        "type": "integer",
        "description": "Size of local heap to reserve in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_size_of_heap_commit": {
        "type": "integer",
        "description": "Size of local heap to commit in bytes.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_number_of_rva_and_sizes": {
        "type": "integer",
        "description": "Number of data directory entries in the optional header.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_section_alignment": {
        "type": "integer",
        "description": "Alignment factor (in bytes) used to align sections in memory.",
        "min": 0,
        "missing_policy": "error",
    },
    "optional_header_file_alignment": {
        "type": "integer",
        "description": "Alignment factor (in bytes) used to align raw data of sections on disk.",
        "min": 0,
        "missing_policy": "error",
    },
    "dll_characteristics_flags": {
        "type": "dict",
        "description": "Interpreted boolean flags for DLL characteristics (ASLR, DEP, SafeSEH, CFG).",
        "missing_policy": "error",
    },

    # Section Information
    "number_of_sections": {
        "type": "integer",
        "description": "Actual number of parsed sections in the PE file.",
        "min": 0,
        "missing_policy": "error",
    },
    "sections": {
        "type": "list",
        "description": "List of dictionaries detailing each PE section.",
        "missing_policy": "error",
        "items": {
            "type": "dict",
            "schema": {
                "index": {"type": "integer", "min": 0},
                "name": {"type": "string"},
                "virtual_address": {"type": "integer", "min": 0},
                "virtual_size": {"type": "integer", "min": 0},
                "size_of_raw_data": {"type": "integer", "min": 0},
                "pointer_to_raw_data": {"type": "integer", "min": 0},
                "entropy": {"type": "float", "min": 0.0, "max": 8.0},
                "characteristics": {"type": "integer", "min": 0},
                "permissions": {"type": "dict"},
            }
        }
    },

    # Import Information
    "number_of_imported_dlls": {
        "type": "integer",
        "description": "Total number of unique DLLs imported by the binary.",
        "min": 0,
        "missing_policy": "error",
    },
    "number_of_imported_functions": {
        "type": "integer",
        "description": "Total number of imported functions across all imported DLLs.",
        "min": 0,
        "missing_policy": "error",
    },
    "imported_dll_names": {
        "type": "list",
        "description": "List of imported DLL module names.",
        "missing_policy": "error",
    },
    "imports": {
        "type": "list",
        "description": "Detailed list of imported DLLs and imported function metadata.",
        "missing_policy": "error",
    },
    "unusual_import_characteristics": {
        "type": "dict",
        "description": "Indicators of unusual import configurations (e.g. ordinal-only imports, suspicious APIs).",
        "missing_policy": "error",
    },

    # Export Information
    "number_of_exports": {
        "type": "integer",
        "description": "Total number of exported functions.",
        "min": 0,
        "missing_policy": "error",
    },
    "export_names": {
        "type": "list",
        "description": "List of exported symbol names.",
        "missing_policy": "error",
    },
    "exports": {
        "type": "list",
        "description": "Detailed list of exported functions (names, ordinals, addresses).",
        "missing_policy": "error",
    },

    # Resource Information
    "has_resources": {
        "type": "boolean",
        "description": "Whether the executable contains a resource directory.",
        "missing_policy": "error",
    },
    "resource_count": {
        "type": "integer",
        "description": "Total number of resource leaf entries.",
        "min": 0,
        "missing_policy": "error",
    },
    "resource_types": {
        "type": "list",
        "description": "List of unique resource types present (e.g. RT_VERSION, RT_MANIFEST).",
        "missing_policy": "error",
    },
    "resources": {
        "type": "list",
        "description": "List of resource entry metadata objects.",
        "missing_policy": "error",
    },

    # Additional Characteristics
    "overlay_size": {
        "type": "integer",
        "description": "Size in bytes of extraneous overlay data appended after the last section.",
        "min": 0,
        "missing_policy": "error",
    },
    "has_overlay": {
        "type": "boolean",
        "description": "Flag indicating presence of overlay data.",
        "missing_policy": "error",
    },
    "overlay_entropy": {
        "type": "float",
        "description": "Shannon entropy of overlay data (0.0 if no overlay).",
        "min": 0.0,
        "max": 8.0,
        "missing_policy": "error",
    },
    "certificate_table": {
        "type": "dict",
        "description": "Security certificate directory details (presence, virtual address, size).",
        "missing_policy": "error",
    },
    "debug_entries_count": {
        "type": "integer",
        "description": "Number of debug directory entries.",
        "min": 0,
        "missing_policy": "error",
    },
    "has_debug_data": {
        "type": "boolean",
        "description": "Flag indicating presence of debug directory information.",
        "missing_policy": "error",
    },
    "debug_types": {
        "type": "list",
        "description": "List of debug information type names.",
        "missing_policy": "error",
    },
}

# Canonical feature ordering
FEATURE_ORDER: List[str] = list(PE_FEATURE_SCHEMA.keys())


def get_feature_order() -> List[str]:
    """Return the ordered list of canonical feature names."""
    return list(FEATURE_ORDER)


def _check_type(val: Any, expected_type: Union[str, List[str]]) -> bool:
    """Check if value matches the expected type specification."""
    if isinstance(expected_type, list):
        return any(_check_type(val, t) for t in expected_type)

    if expected_type == "integer":
        # In Python, bool is a subclass of int; reject bool when expecting int
        return isinstance(val, int) and not isinstance(val, bool)
    elif expected_type == "float":
        return isinstance(val, (float, int)) and not isinstance(val, bool)
    elif expected_type == "string":
        return isinstance(val, str)
    elif expected_type == "boolean":
        return isinstance(val, bool)
    elif expected_type == "list":
        return isinstance(val, list)
    elif expected_type == "dict":
        return isinstance(val, dict)
    elif expected_type == "null":
        return val is None
    return True


def validate_feature_schema(
    features: Dict[str, Any],
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate an extracted feature dictionary against the PE_FEATURE_SCHEMA.

    Args:
        features: Feature dictionary to validate.
        strict: If True, reject any unexpected feature keys not defined in the schema.

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []

    # Check schema version
    extracted_version = features.get("schema_version")
    if extracted_version != SCHEMA_VERSION:
        errors.append(
            f"Schema version mismatch: expected '{SCHEMA_VERSION}', got '{extracted_version}'"
        )

    # Check strict mode for unexpected keys
    if strict:
        schema_keys = set(PE_FEATURE_SCHEMA.keys())
        extra_keys = set(features.keys()) - schema_keys
        if extra_keys:
            errors.append(f"Unexpected features found in strict mode: {sorted(list(extra_keys))}")

    # Validate each schema specification
    for name, spec in PE_FEATURE_SCHEMA.items():
        if name not in features:
            if spec.get("missing_policy") == "error":
                errors.append(f"Missing required feature: '{name}'")
            continue

        val = features[name]
        expected_type = spec.get("type")

        # Type checking
        if expected_type is not None and not _check_type(val, expected_type):
            errors.append(
                f"Feature '{name}' type mismatch: expected {expected_type}, got {type(val).__name__}"
            )
            continue

        # Range checks for numeric values
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            min_val = spec.get("min")
            if min_val is not None and val < min_val:
                errors.append(f"Feature '{name}' value {val} is less than min {min_val}")

            max_val = spec.get("max")
            if max_val is not None and val > max_val:
                errors.append(f"Feature '{name}' value {val} exceeds max {max_val}")

        # Allowed values check
        allowed = spec.get("values")
        if allowed is not None:
            if isinstance(allowed, list) and val not in allowed:
                errors.append(f"Feature '{name}' value {val} not in allowed values: {allowed}")
            elif isinstance(allowed, dict) and val not in allowed:
                errors.append(f"Feature '{name}' value {val} not in allowed values: {list(allowed.keys())}")

        # Nested section item validation
        if name == "sections" and isinstance(val, list):
            item_schema = spec.get("items", {}).get("schema", {})
            for idx, section_dict in enumerate(val):
                if not isinstance(section_dict, dict):
                    errors.append(f"Section index {idx} is not a dictionary")
                    continue
                for sec_key, sec_spec in item_schema.items():
                    if sec_key in section_dict:
                        sec_val = section_dict[sec_key]
                        if not _check_type(sec_val, sec_spec.get("type")):
                            errors.append(
                                f"Section {idx} key '{sec_key}' type mismatch: "
                                f"expected {sec_spec.get('type')}, got {type(sec_val).__name__}"
                            )
                        if isinstance(sec_val, (int, float)) and not isinstance(sec_val, bool):
                            if "min" in sec_spec and sec_val < sec_spec["min"]:
                                errors.append(f"Section {idx} key '{sec_key}' value below min")
                            if "max" in sec_spec and sec_val > sec_spec["max"]:
                                errors.append(f"Section {idx} key '{sec_key}' value above max")

    is_valid = len(errors) == 0
    if not is_valid:
        for err in errors:
            logger.warning(f"Schema validation error: {err}")

    return is_valid, errors


def validate_features(
    features: Dict[str, Any],
    strict: bool = False,
    raise_on_error: bool = False,
) -> bool:
    """
    Convenience function to validate features against PE_FEATURE_SCHEMA.

    Args:
        features: Feature dictionary.
        strict: Reject unexpected feature keys if True.
        raise_on_error: Raise SchemaValidationError if validation fails.

    Returns:
        True if valid, False otherwise.
    """
    is_valid, errors = validate_feature_schema(features, strict=strict)
    if not is_valid and raise_on_error:
        raise SchemaValidationError(f"Feature schema validation failed with {len(errors)} error(s): {errors}")
    return is_valid


__all__ = [
    "SCHEMA_VERSION",
    "PE_FEATURE_SCHEMA",
    "FEATURE_ORDER",
    "FeatureSpec",
    "SchemaValidationError",
    "get_feature_order",
    "validate_feature_schema",
    "validate_features",
]
