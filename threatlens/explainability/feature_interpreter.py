"""
ThreatLens Feature Interpretation Layer

Maps raw EMBER 2018 feature indices to canonical feature definitions, schema groups,
technical descriptions, SHAP attribution values, influence directions, and explicit
interpretation limitations.

CRITICAL POLICY:
Hashed features (such as 2D MurmurHash import/export/section bins) must NOT have
invented semantic meanings. Their descriptions and limitations must explicitly state
that they are feature-hashed representations subject to hash collisions and one-way
hash mapping.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from threatlens.config import FEATURE_GROUPS, EmberFeatureGroups
from threatlens.utils.logger import get_logger

logger = get_logger("feature_interpreter")


# Enum-like direction constants
DIRECTION_INCREASES_MALWARE = "INCREASES_MALWARE_RISK"
DIRECTION_DECREASES_MALWARE = "DECREASES_MALWARE_RISK"
DIRECTION_NEUTRAL = "NEUTRAL"


@dataclass
class FeatureContribution:
    """
    Detailed attribution and interpretation for an individual feature.
    """
    feature_index: int
    feature_name: str
    feature_group: str
    feature_description: str
    feature_value: float
    shap_value: float
    abs_shap_value: float
    direction: str
    interpretation_limitations: str
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert contribution to JSON-serializable dictionary."""
        return {
            "feature_index": int(self.feature_index),
            "feature_name": str(self.feature_name),
            "feature_group": str(self.feature_group),
            "feature_description": str(self.feature_description),
            "feature_value": float(self.feature_value),
            "shap_value": float(self.shap_value),
            "abs_shap_value": float(self.abs_shap_value),
            "direction": str(self.direction),
            "interpretation_limitations": str(self.interpretation_limitations),
            "rank": int(self.rank) if self.rank is not None else None,
        }


# Canonical definitions for known scalar features in the EMBER 2018 schema
_GENERAL_FEATURE_METADATA: Dict[int, Dict[str, str]] = {
    616: {
        "name": "ember_general_size",
        "label": "file_size",
        "description": "Total physical binary file size on disk in bytes.",
        "limitations": "File size alone is non-specific; large files may indicate bundling or overlay padding, while small files may indicate droppers or stagers.",
    },
    617: {
        "name": "ember_general_vsize",
        "label": "virtual_size",
        "description": "Allocated virtual memory size (SizeOfImage) when mapped into process memory.",
        "limitations": "Discrepancies between disk size and virtual size often indicate packing or runtime expansion, but standard packers and self-extracting installers also exhibit this.",
    },
    618: {
        "name": "ember_general_has_debug",
        "label": "has_debug",
        "description": "Presence of PE debug directory: 1 if debug symbols/directory present, 0 if stripped.",
        "limitations": "Stripped debug information is standard in production commercial binaries as well as malware.",
    },
    619: {
        "name": "ember_general_exports",
        "label": "exports_count",
        "description": "Total number of exported functions/symbols in the PE export table.",
        "limitations": "High export counts are expected in dynamic link libraries (DLLs); zero exports are standard for stand-alone executables.",
    },
    620: {
        "name": "ember_general_imports",
        "label": "imports_count",
        "description": "Total number of imported functions/APIs referenced across all imported DLLs.",
        "limitations": "Abnormally low import counts suggest runtime API resolution (e.g. LoadLibrary/GetProcAddress) or packing, but could also reflect statically linked minimal binaries.",
    },
    621: {
        "name": "ember_general_has_relocations",
        "label": "has_relocations",
        "description": "Presence of base relocations section: 1 if present (.reloc), 0 if absent.",
        "limitations": "Absence of relocations is common in older non-ASLR 32-bit executables and stripped malware.",
    },
    622: {
        "name": "ember_general_has_resources",
        "label": "has_resources",
        "description": "Presence of PE resource directory (.rsrc): 1 if present, 0 if absent.",
        "limitations": "Legitimate applications commonly bundle UI resources; malware frequently stages encrypted payloads or decoy icons in resources.",
    },
    623: {
        "name": "ember_general_has_signature",
        "label": "has_signature",
        "description": "Presence of Authenticode digital certificate directory: 1 if signed, 0 if unsigned.",
        "limitations": "Unsigned binaries are widespread in open-source and internal software; signatures may also be self-signed, expired, or stolen.",
    },
    624: {
        "name": "ember_general_has_tls",
        "label": "has_tls",
        "description": "Presence of Thread Local Storage (TLS) directory: 1 if present, 0 if absent.",
        "limitations": "TLS callbacks allow code execution before the main entry point (often leveraged for evasion), but TLS is also standard in multithreaded CRT runtimes.",
    },
    625: {
        "name": "ember_general_symbols",
        "label": "symbols_count",
        "description": "Number of symbol table entries in the COFF symbol table.",
        "limitations": "COFF symbol tables are virtually always stripped (0) in modern Windows production PE binaries.",
    },
    # Also support legacy synthetic index mappings for backward compatibility
    768: {"name": "ember_general_size", "label": "file_size", "description": "Total physical binary file size on disk in bytes.", "limitations": "File size alone is non-specific."},
    769: {"name": "ember_general_vsize", "label": "virtual_size", "description": "Allocated virtual memory size when mapped.", "limitations": "Discrepancies often indicate packing."},
    770: {"name": "ember_general_has_debug", "label": "has_debug", "description": "Presence of PE debug directory.", "limitations": "Stripped debug info is standard."},
    771: {"name": "ember_general_exports", "label": "exports_count", "description": "Total number of exported functions.", "limitations": "Expected in DLLs."},
    772: {"name": "ember_general_imports", "label": "imports_count", "description": "Total number of imported functions.", "limitations": "Low imports suggest packing."},
    773: {"name": "ember_general_has_relocations", "label": "has_relocations", "description": "Presence of base relocations section.", "limitations": "Common in non-ASLR."},
    774: {"name": "ember_general_has_resources", "label": "has_resources", "description": "Presence of PE resource directory.", "limitations": "Bundled in legitimate UI & malware."},
    775: {"name": "ember_general_has_signature", "label": "has_signature", "description": "Presence of Authenticode digital certificate.", "limitations": "Unsigned binaries widespread."},
    776: {"name": "ember_general_has_tls", "label": "has_tls", "description": "Presence of Thread Local Storage (TLS) directory.", "limitations": "TLS callbacks allow pre-main execution."},
    777: {"name": "ember_general_symbols", "label": "symbols_count", "description": "Number of symbol table entries in COFF symbol table.", "limitations": "Usually stripped in release."},
}

_STRINGS_SCALAR_METADATA: Dict[int, Dict[str, str]] = {
    512: {
        "name": "ember_strings_numstrings",
        "description": "Total count of extracted printable ASCII strings (minimum length 5).",
        "limitations": "Low string counts strongly correlate with packing or string encryption; legitimate compiled tools vary widely.",
    },
    513: {
        "name": "ember_strings_avlength",
        "description": "Average character length of extracted printable strings.",
        "limitations": "Statistical aggregate; does not reveal whether strings are human-readable text, URLs, or base64 blobs.",
    },
    514: {
        "name": "ember_strings_printables",
        "description": "Total count of printable ASCII characters across all extracted strings.",
        "limitations": "Volume metric correlated with overall binary size and string density.",
    },
    611: {
        "name": "ember_strings_entropy",
        "description": "Information entropy across extracted printable character distribution.",
        "limitations": "Reflects character diversity, but cannot distinguish encoded payloads from natural multi-language text.",
    },
    612: {
        "name": "ember_strings_paths",
        "description": "Count of file path patterns (e.g. 'C:\\', '/usr/') identified in strings.",
        "limitations": "Heuristic regex match; subject to false positives from binary sequences resembling file paths.",
    },
    613: {
        "name": "ember_strings_urls",
        "description": "Count of URL/URI patterns (e.g. 'http://', 'https://') identified in strings.",
        "limitations": "Heuristic regex match; detects network references but does not verify active C2 communication.",
    },
    614: {
        "name": "ember_strings_registry",
        "description": "Count of Windows Registry key references (e.g. 'HKEY_', 'Software\\') in strings.",
        "limitations": "Registry access is standard across legitimate Windows software as well as persistence malware.",
    },
    615: {
        "name": "ember_strings_mz",
        "description": "Count of embedded 'MZ' PE header magic signatures found within string data.",
        "limitations": "Embedded MZ headers often indicate bundled sub-executables or payload dropping, but can also occur in installers or self-extracting zip archives.",
    },
}

_HEADER_SCALAR_METADATA: Dict[int, Dict[str, str]] = {
    626: {
        "name": "ember_header_timestamp",
        "description": "COFF compilation timestamp (TimeDateStamp) in 32-bit Unix epoch seconds.",
        "limitations": "Timestamps can be easily spoofed (timestomping) or zeroed out during compilation/linking.",
    },
    677: {
        "name": "ember_header_major_image_version",
        "description": "Major image version number from optional header.",
        "limitations": "Developer-specified version metadata; easily modified.",
    },
    678: {
        "name": "ember_header_minor_image_version",
        "description": "Minor image version number from optional header.",
        "limitations": "Developer-specified version metadata; easily modified.",
    },
    679: {
        "name": "ember_header_major_linker_version",
        "description": "Major linker version number identifying compiler/linker toolchain.",
        "limitations": "Indicates build toolchain age; can be forged or altered.",
    },
    680: {
        "name": "ember_header_minor_linker_version",
        "description": "Minor linker version number identifying compiler/linker toolchain.",
        "limitations": "Toolchain version detail.",
    },
    685: {
        "name": "ember_header_sizeof_code",
        "description": "Size of the executable code section (.text) in bytes.",
        "limitations": "Correlated with binary complexity; small sizes can indicate stagers.",
    },
    686: {
        "name": "ember_header_sizeof_headers",
        "description": "Total size of all PE headers rounded up to file alignment.",
        "limitations": "Structural PE format field.",
    },
    # Also support legacy indices
    778: {
        "name": "ember_header_timestamp",
        "description": "COFF compilation timestamp (TimeDateStamp) in 32-bit Unix epoch seconds.",
        "limitations": "Timestamps can be easily spoofed (timestomping) or zeroed out during compilation/linking.",
    },
    784: {
        "name": "ember_header_characteristics",
        "description": "COFF header characteristics flags bitmask (executable, 32-bit, DLL, etc.).",
        "limitations": "Binary flags describing compilation properties.",
    },
    798: {
        "name": "ember_header_optional_magic",
        "description": "Optional header magic word (0x10B for PE32 executable, 0x20B for PE32+ 64-bit).",
        "limitations": "Standard PE architecture discriminant.",
    },
    799: {
        "name": "ember_header_major_linker_version",
        "description": "Major version of the linker used to produce the executable.",
        "limitations": "Indicates compiler toolchain vintage; can be forged or modified.",
    },
    800: {
        "name": "ember_header_minor_linker_version",
        "description": "Minor version of the linker used to produce the executable.",
        "limitations": "Toolchain metadata.",
    },
    801: {
        "name": "ember_header_size_of_code",
        "description": "Size of the executable code section (.text) in bytes.",
        "limitations": "Reflects compiled code volume; small code sizes with large virtual sizes suggest packing.",
    },
    802: {
        "name": "ember_header_size_of_initialized_data",
        "description": "Size of initialized data sections in bytes.",
        "limitations": "Volume metric for embedded global data and constants.",
    },
    803: {
        "name": "ember_header_size_of_uninitialized_data",
        "description": "Size of uninitialized data sections (.bss) in bytes.",
        "limitations": "Volume metric for runtime-allocated uninitialized storage.",
    },
    804: {
        "name": "ember_header_address_of_entry_point",
        "description": "Relative Virtual Address (RVA) of the entry point where execution begins.",
        "limitations": "Entry points outside standard .text sections frequently indicate packers or protectors.",
    },
    805: {
        "name": "ember_header_base_of_code",
        "description": "RVA of the beginning of the code section.",
        "limitations": "Standard PE memory layout property.",
    },
    806: {
        "name": "ember_header_base_of_data",
        "description": "RVA of the beginning of the data section (PE32 only).",
        "limitations": "Standard PE memory layout property.",
    },
    807: {
        "name": "ember_header_image_base",
        "description": "Preferred base address for loading the executable into memory.",
        "limitations": "Standard default is 0x400000 for EXE or 0x10000000 for DLL, subject to ASLR rebasing.",
    },
    808: {
        "name": "ember_header_section_alignment",
        "description": "Alignment of sections when mapped into virtual memory (typically 4096 bytes).",
        "limitations": "PE memory page granularity parameter.",
    },
    809: {
        "name": "ember_header_file_alignment",
        "description": "Alignment of sections on disk (typically 512 bytes).",
        "limitations": "PE disk sector granularity parameter.",
    },
    810: {
        "name": "ember_header_major_os_version",
        "description": "Major operating system version required to run this binary.",
        "limitations": "Compatibility manifest field.",
    },
    811: {
        "name": "ember_header_minor_os_version",
        "description": "Minor operating system version required to run this binary.",
        "limitations": "Compatibility manifest field.",
    },
}


class FeatureInterpreter:
    """
    Feature interpretation layer for ThreatLens EMBER 2018 models.
    Provides canonical metadata, human-readable descriptions, SHAP directional
    interpretations, and explicit limitation disclosures for all 2,351 features.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize feature interpreter.

        Args:
            feature_names: Optional list of 2,351 feature names. If None,
                           generates canonical EMBER names.
        """
        self.expected_dim = EmberFeatureGroups.TOTAL_FEATURES  # 2351
        if feature_names is not None:
            if len(feature_names) != self.expected_dim:
                logger.warning(
                    f"Feature names list length ({len(feature_names)}) != {self.expected_dim}. "
                    "Falling back to canonical EMBER feature names."
                )
                self.feature_names = self._generate_canonical_names()
            else:
                self.feature_names = list(feature_names)
        else:
            self.feature_names = self._generate_canonical_names()

    @staticmethod
    def _generate_canonical_names() -> List[str]:
        """Generate canonical 2,351 EMBER feature names."""
        names = []
        for group in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(group)
            for i in range(start, end):
                names.append(f"ember_{group}_{i}")
        return names

    def get_feature_group(self, feature_index: int) -> str:
        """
        Determine the official EMBER feature group for a given index.

        Args:
            feature_index: Integer index in [0, 2350]

        Returns:
            Group name string
        """
        if not (0 <= feature_index < self.expected_dim):
            return "unknown"

        for group in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(group)
            if start <= feature_index < end:
                return group

        return "unknown"

    def get_feature_name(self, feature_index: int) -> str:
        """Get canonical or registered name for a feature index."""
        group = self.get_feature_group(feature_index)
        if 0 <= feature_index < len(self.feature_names):
            name = self.feature_names[feature_index]
            # Ensure name matches current group schema; if legacy model had mismatched group, use canonical
            if group != "unknown" and group not in name and not name.startswith("Column_"):
                return f"ember_{group}_{feature_index}"
            return name
        return f"ember_{group}_{feature_index}" if group != "unknown" else f"ember_feature_{feature_index}"

    def get_feature_description(
        self, feature_index: int, feature_name: Optional[str] = None
    ) -> str:
        """
        Generate detailed, technically rigorous feature description.
        Adheres to policy of NOT inventing semantic meanings for hashed features.
        """
        group = self.get_feature_group(feature_index)

        # 1. Byte Histogram: 0 - 255
        if group == "histogram":
            byte_val = feature_index
            return (
                f"Normalized frequency of raw byte 0x{byte_val:02X} ({byte_val}) "
                f"across the binary file."
            )

        # 2. Byte-Entropy Histogram: 256 - 511
        if group == "byteentropy":
            start, _ = FEATURE_GROUPS.get_group_indices("byteentropy")
            bin_idx = feature_index - start
            return (
                f"2D Byte-entropy histogram joint distribution bin {bin_idx} "
                "(measures localized Shannon entropy versus byte values across sliding windows)."
            )

        # 3. String features
        if group == "strings":
            if feature_index in _STRINGS_SCALAR_METADATA:
                return _STRINGS_SCALAR_METADATA[feature_index]["description"]
            start, _ = FEATURE_GROUPS.get_group_indices("strings")
            bin_idx = feature_index - start
            return (
                f"Printable string character/length distribution bin {bin_idx} "
                "(frequency count across string character classes)."
            )

        # 4. General file info
        if group == "general":
            if feature_index in _GENERAL_FEATURE_METADATA:
                return _GENERAL_FEATURE_METADATA[feature_index]["description"]
            return f"General PE binary header metadata attribute (index {feature_index})."

        # 5. Header features
        if group == "header":
            if feature_index in _HEADER_SCALAR_METADATA:
                return _HEADER_SCALAR_METADATA[feature_index]["description"]
            start, _ = FEATURE_GROUPS.get_group_indices("header")
            offset = feature_index - start
            return (
                f"COFF or Optional PE header field / subsystem flag attribute (offset {offset})."
            )

        # 6. Section characteristics
        if group == "section":
            start, _ = FEATURE_GROUPS.get_group_indices("section")
            sec_idx = feature_index - start
            return (
                f"Section characteristics feature {sec_idx} "
                "(encodes section sizes, virtual memory entropy, permissions, and hashed section names)."
            )

        # 7. Imports
        if group == "imports":
            start, _ = FEATURE_GROUPS.get_group_indices("imports")
            bin_idx = feature_index - start
            return (
                f"Feature-hashed import representation (bin index {bin_idx}). "
                "Encodes imported dynamic link libraries (DLLs) and API function symbols via MurmurHash."
            )

        # 8. Exports
        if group == "exports":
            start, _ = FEATURE_GROUPS.get_group_indices("exports")
            bin_idx = feature_index - start
            return (
                f"Feature-hashed export representation (bin index {bin_idx}). "
                "Encodes exported function symbols from the PE export directory via MurmurHash."
            )

        return f"Unknown feature index {feature_index}."

    def get_feature_limitations(
        self, feature_index: int, feature_group: Optional[str] = None
    ) -> str:
        """
        Return technical and interpretive limitations for the feature.
        """
        group = feature_group or self.get_feature_group(feature_index)

        if group == "histogram":
            return (
                "Raw byte frequency indicates statistical composition (e.g. zero-padding, compression, "
                "or encryption), but does not reveal opcode semantics, control flow, or malicious intent."
            )

        if group == "byteentropy":
            return (
                "Reflects localized statistical randomness patterns (e.g. distinguishing plaintext code from "
                "packed/encrypted payloads), but cannot identify specific cryptographic algorithms or code blocks."
            )

        if group == "strings":
            if feature_index in _STRINGS_SCALAR_METADATA:
                return _STRINGS_SCALAR_METADATA[feature_index]["limitations"]
            return (
                "Aggregated string histogram bin; provides statistical character distribution but does not "
                "capture individual string text, syntax, or runtime context."
            )

        if group == "general":
            if feature_index in _GENERAL_FEATURE_METADATA:
                return _GENERAL_FEATURE_METADATA[feature_index]["limitations"]
            return "General PE file metadata can be manipulated by packers and benign compilers alike."

        if group == "header":
            if feature_index in _HEADER_SCALAR_METADATA:
                return _HEADER_SCALAR_METADATA[feature_index]["limitations"]
            return (
                "PE header fields can be modified by linkers, protectors, or timestomping tools without "
                "altering the underlying executable payload behavior."
            )

        if group == "section":
            return (
                "Section layout and permissions reflect compilation settings and packing; hashed section "
                "name bins cannot be uniquely mapped back to raw section names due to hash collisions."
            )

        if group == "imports":
            return (
                "EMBER uses 2D MurmurHash projection across imported DLLs and function names. Due to hash "
                "collisions and one-way hashing, this bin CANNOT be uniquely attributed to a specific API call."
            )

        if group == "exports":
            return (
                "Feature-hashed export bin; the underlying exported function symbol name CANNOT be uniquely "
                "recovered due to one-way hash mapping and collision possibilities."
            )

        return "Feature interpretation limitations unspecified for unknown feature."

    def interpret_contribution(
        self,
        feature_index: int,
        feature_value: float,
        shap_value: float,
        feature_name: Optional[str] = None,
        rank: Optional[int] = None,
    ) -> FeatureContribution:
        """
        Construct a complete, validated FeatureContribution object.

        Args:
            feature_index: Numerical index [0, 2350]
            feature_value: Observed feature value in sample
            shap_value: Signed SHAP contribution value phi_i
            feature_name: Optional feature name (defaults to canonical name)
            rank: Optional rank by absolute contribution

        Returns:
            Populated FeatureContribution dataclass
        """
        canonical_name = feature_name or self.get_feature_name(feature_index)
        group = self.get_feature_group(feature_index)
        description = self.get_feature_description(feature_index, canonical_name)
        limitations = self.get_feature_limitations(feature_index, group)

        # Direction determination for binary classification where positive class = 1 (Malicious)
        abs_val = abs(float(shap_value))
        if shap_value > 1e-7:
            direction = DIRECTION_INCREASES_MALWARE
        elif shap_value < -1e-7:
            direction = DIRECTION_DECREASES_MALWARE
        else:
            direction = DIRECTION_NEUTRAL

        return FeatureContribution(
            feature_index=int(feature_index),
            feature_name=canonical_name,
            feature_group=group,
            feature_description=description,
            feature_value=float(feature_value),
            shap_value=float(shap_value),
            abs_shap_value=abs_val,
            direction=direction,
            interpretation_limitations=limitations,
            rank=int(rank) if rank is not None else None,
        )

    def interpret_vector(
        self,
        feature_vector: np.ndarray,
        shap_values: np.ndarray,
    ) -> List[FeatureContribution]:
        """
        Interpret all features for a single sample.

        Args:
            feature_vector: 1D array of shape (2351,)
            shap_values: 1D array of shape (2351,)

        Returns:
            List of 2,351 FeatureContribution instances
        """
        if len(feature_vector) != self.expected_dim or len(shap_values) != self.expected_dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.expected_dim} features, "
                f"got {len(feature_vector)} feature values and {len(shap_values)} SHAP values."
            )

        contributions: List[FeatureContribution] = []
        for i in range(self.expected_dim):
            contrib = self.interpret_contribution(
                feature_index=i,
                feature_value=float(feature_vector[i]),
                shap_value=float(shap_values[i]),
            )
            contributions.append(contrib)

        return contributions


__all__ = [
    "FeatureContribution",
    "FeatureInterpreter",
    "DIRECTION_INCREASES_MALWARE",
    "DIRECTION_DECREASES_MALWARE",
    "DIRECTION_NEUTRAL",
]
