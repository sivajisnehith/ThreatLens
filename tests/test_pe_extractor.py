"""
Unit Tests for PEFeatureExtractor

Tests static PE analysis on valid, invalid, malformed, empty, and truncated files.
Guarantees deterministic execution and that no binary is ever executed.
"""

from pathlib import Path
import struct
import sys
import tempfile
import pytest

from threatlens.core.extractor import (
    PEFeatureExtractor,
    PEParseError,
    calculate_entropy,
    extract_pe_features,
)
from threatlens.core.schema import SCHEMA_VERSION


def create_synthetic_pe(
    num_sections: int = 1,
    has_imports: bool = True,
    has_exports: bool = False,
    section_entropy_data: bytes = b"\x90" * 512,
    overlay_data: bytes = b"",
    corrupt_dos_magic: bool = False,
    corrupt_lfanew: bool = False,
    corrupt_nt_signature: bool = False,
) -> bytes:
    """
    Construct a deterministic in-memory synthetic PE binary.
    Does not contain any executable payload or malware.
    """
    dos_header = bytearray(64)
    dos_header[0:2] = b"XX" if corrupt_dos_magic else b"MZ"

    lfanew = 128 if corrupt_lfanew else 64
    struct.pack_into("<I", dos_header, 60, lfanew)

    # NT Headers
    pe_header = bytearray()
    pe_header.extend(b"XX\x00\x00" if corrupt_nt_signature else b"PE\x00\x00")

    # COFF File Header (20 bytes)
    # Machine=i386 (0x14C), NumberOfSections, TimeDateStamp=0x60000000, PointerToSymbolTable=0,
    # NumberOfSymbols=0, SizeOfOptionalHeader=224, Characteristics=0x0102 (EXECUTABLE_IMAGE | 32BIT_MACHINE)
    pe_header.extend(struct.pack("<HHIIIHH", 0x14C, num_sections, 0x60000000, 0, 0, 224, 0x0102))

    # Optional Header Standard Fields (28 bytes)
    # Magic=0x10B (PE32), MajorLinker=14, MinorLinker=0, SizeOfCode=0x1000, SizeOfInitData=0x1000,
    # SizeOfUninitData=0, AddressOfEntryPoint=0x1000, BaseOfCode=0x1000, BaseOfData=0x2000
    pe_header.extend(struct.pack("<HBBIIIIII", 0x10B, 14, 0, 0x1000, 0x1000, 0, 0x1000, 0x1000, 0x2000))

    # Optional Header Windows-Specific Fields (44 bytes)
    # ImageBase=0x400000, SectionAlignment=0x1000, FileAlignment=0x200, OS/Subsys versions,
    # Win32Version=0, SizeOfImage=0x10000, SizeOfHeaders=0x400, CheckSum=0x1234, Subsystem=2 (GUI), DllCharacteristics=0x8140
    pe_header.extend(struct.pack("<IIIHHHHHHIIIIHH", 0x400000, 0x1000, 0x200, 4, 0, 0, 0, 4, 0, 0, 0x10000, 0x400, 0x1234, 2, 0x8140))

    # Stack & Heap sizes, LoaderFlags, NumberOfRvaAndSizes=16 (32 bytes)
    pe_header.extend(struct.pack("<IIIIII", 0x100000, 0x1000, 0x100000, 0x1000, 0, 16))

    # 16 Data Directories (16 * 8 = 128 bytes)
    data_dirs = bytearray(128)
    if has_exports:
        # Export directory at RVA 0x3000, size 0x100
        struct.pack_into("<II", data_dirs, 0, 0x3000, 0x100)
    if has_imports:
        # Import directory at RVA 0x2000, size 0x100
        struct.pack_into("<II", data_dirs, 8, 0x2000, 0x100)

    pe_header.extend(data_dirs)

    # Section Headers (40 bytes each)
    section_headers = bytearray()
    raw_offset = 0x400
    file_alignment = 0x200

    section_names = [b".text\x00\x00\x00", b".data\x00\x00\x00", b".rsrc\x00\x00\x00", b".reloc\x00\x00"]
    characteristics_map = [
        0x60000020,  # .text: CODE | EXECUTE | READ
        0xC0000040,  # .data: INITIALIZED_DATA | READ | WRITE
        0x40000040,  # .rsrc: INITIALIZED_DATA | READ
        0x42000040,  # .reloc: INITIALIZED_DATA | DISCARDABLE | READ
    ]

    for i in range(num_sections):
        sec_name = section_names[i % len(section_names)]
        sec_chars = characteristics_map[i % len(characteristics_map)]
        sec_hdr = bytearray(40)
        sec_hdr[0:8] = sec_name
        # VirtualSize=0x1000, VirtualAddress, SizeOfRawData=len(section_entropy_data), PointerToRawData
        struct.pack_into(
            "<IIIIIIHHI",
            sec_hdr,
            8,
            0x1000,
            0x1000 * (i + 1),
            len(section_entropy_data),
            raw_offset,
            0,
            0,
            0,
            0,
            sec_chars,
        )
        section_headers.extend(sec_hdr)
        raw_offset += len(section_entropy_data)

    # Assemble complete buffer
    header_data = bytes(dos_header) + bytes(pe_header) + bytes(section_headers)
    padded_headers = header_data.ljust(0x400, b"\x00")

    body = bytearray()
    for _ in range(num_sections):
        body.extend(section_entropy_data)

    return padded_headers + bytes(body) + overlay_data


class TestPEFeatureExtractor:
    """Comprehensive test suite for static PE feature extraction."""

    def test_valid_synthetic_pe(self):
        """Test extraction on a valid synthetic PE."""
        pe_bytes = create_synthetic_pe(num_sections=1)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()

        assert features["schema_version"] == SCHEMA_VERSION
        assert features["file_type"] == "PE32"
        assert features["machine_architecture"] == 0x14C
        assert features["number_of_sections"] == 1
        assert len(features["sections"]) == 1
        assert features["sections"][0]["name"] == ".text"
        assert features["dos_header_magic"] == 0x5A4D
        assert features["has_overlay"] is False

    def test_benign_system_executable(self):
        """Test extraction on known benign host binary (python.exe)."""
        extractor = PEFeatureExtractor(sys.executable)
        features = extractor.extract()

        assert features["schema_version"] == SCHEMA_VERSION
        assert features["file_type"] in ["PE32", "PE32+"]
        assert features["file_size"] > 0
        assert features["number_of_sections"] > 0
        assert features["number_of_imported_dlls"] > 0
        assert "dos_header" in features
        assert "characteristics" in features

    def test_invalid_executable_magic(self):
        """Test that invalid DOS magic raises PEParseError."""
        invalid_bytes = b"NOT_A_PE_FILE_HEADER_DATA_1234567890"
        with pytest.raises(PEParseError) as exc_info:
            PEFeatureExtractor(data=invalid_bytes)
        assert "Invalid PE format" in str(exc_info.value) or "magic" in str(exc_info.value).lower()

    def test_empty_buffer_and_file(self):
        """Test that empty bytes or 0-byte file raises PEParseError."""
        with pytest.raises(PEParseError) as exc_info:
            PEFeatureExtractor(data=b"")
        assert "empty" in str(exc_info.value).lower()

        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with pytest.raises(PEParseError) as exc_info:
                PEFeatureExtractor(pefile_path=tmp_path)
            assert "0 bytes" in str(exc_info.value) or "empty" in str(exc_info.value).lower()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_truncated_pe_file(self):
        """Test that truncated PE headers raise PEParseError."""
        full_pe = create_synthetic_pe(num_sections=2)
        # Truncate midway through the NT header
        truncated_bytes = full_pe[:80]
        with pytest.raises(PEParseError):
            PEFeatureExtractor(data=truncated_bytes)

    def test_malformed_pe_headers(self):
        """Test that invalid e_lfanew or corrupted signature raises PEParseError."""
        corrupt_pe = create_synthetic_pe(corrupt_lfanew=True)
        with pytest.raises(PEParseError):
            PEFeatureExtractor(data=corrupt_pe)

        corrupt_sig_pe = create_synthetic_pe(corrupt_nt_signature=True)
        with pytest.raises(PEParseError):
            PEFeatureExtractor(data=corrupt_sig_pe)

    def test_missing_import_directory(self):
        """Test valid PE without import directory handles gracefully."""
        pe_bytes = create_synthetic_pe(has_imports=False)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()

        assert features["number_of_imported_dlls"] == 0
        assert features["number_of_imported_functions"] == 0
        assert features["imports"] == []
        assert features["imported_dll_names"] == []
        assert features["unusual_import_characteristics"]["has_empty_dll_names"] is False

    def test_missing_export_directory(self):
        """Test valid PE without export directory handles gracefully."""
        pe_bytes = create_synthetic_pe(has_exports=False)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()

        assert features["number_of_exports"] == 0
        assert features["export_names"] == []
        assert features["exports"] == []

    def test_multiple_sections(self):
        """Test extraction across multiple sections."""
        pe_bytes = create_synthetic_pe(num_sections=3)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()

        assert features["number_of_sections"] == 3
        assert len(features["sections"]) == 3
        names = [s["name"] for s in features["sections"]]
        assert ".text" in names
        assert ".data" in names
        assert ".rsrc" in names

    def test_section_entropy_calculation(self):
        """Test entropy calculation with uniform data and high-entropy random data."""
        # Uniform zero data: entropy = 0.0
        uniform_data = b"\x00" * 512
        assert calculate_entropy(uniform_data) == 0.0

        # Uniform single-byte non-zero: entropy = 0.0
        assert calculate_entropy(b"\x90" * 512) == 0.0

        # 2 equally frequent symbols: entropy = 1.0
        two_symbols = b"\x00\xFF" * 256
        assert abs(calculate_entropy(two_symbols) - 1.0) < 1e-4

        # Full 256 byte distribution: entropy = 8.0
        all_bytes = bytes(range(256)) * 4
        assert abs(calculate_entropy(all_bytes) - 8.0) < 1e-4

        # Test on synthetic PE section
        pe_bytes = create_synthetic_pe(num_sections=1, section_entropy_data=all_bytes)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()
        sec_entropy = features["sections"][0]["entropy"]
        assert sec_entropy > 7.5

    def test_section_permissions_extraction(self):
        """Test section permission parsing for readable, writable, executable flags."""
        pe_bytes = create_synthetic_pe(num_sections=2)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()

        text_perms = features["sections"][0]["permissions"]
        assert text_perms["executable"] is True
        assert text_perms["readable"] is True
        assert text_perms["writable"] is False
        assert text_perms["contains_code"] is True

        data_perms = features["sections"][1]["permissions"]
        assert data_perms["executable"] is False
        assert data_perms["readable"] is True
        assert data_perms["writable"] is True
        assert data_perms["contains_initialized_data"] is True

    def test_overlay_extraction(self):
        """Test detection and entropy calculation of overlay data appended after sections."""
        overlay = b"THREATLENS_EXTRA_OVERLAY_BYTES_PAYLOAD" * 10
        pe_bytes = create_synthetic_pe(num_sections=1, overlay_data=overlay)
        extractor = PEFeatureExtractor(data=pe_bytes)
        features = extractor.extract()

        assert features["has_overlay"] is True
        assert features["overlay_size"] == len(overlay)
        assert features["overlay_entropy"] > 0.0

    def test_file_not_found(self):
        """Test that non-existent file path raises PEParseError."""
        fake_path = Path("non_existent_binary_file_12345.exe")
        with pytest.raises(PEParseError) as exc_info:
            PEFeatureExtractor(pefile_path=fake_path)
        assert "not found" in str(exc_info.value).lower()

    def test_convenience_function(self):
        """Test extract_pe_features convenience function."""
        pe_bytes = create_synthetic_pe(num_sections=1)
        features = extract_pe_features(data=pe_bytes)
        assert isinstance(features, dict)
        assert features["file_size"] == len(pe_bytes)
