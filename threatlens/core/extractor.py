"""
ThreatLens PE File Static Feature Extraction Engine

Extracts comprehensive structural and statistical features from Windows PE executables
without executing them, utilizing pefile. Handles malformed, truncated, and missing
PE structures safely.
"""

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pefile

from threatlens.core.schema import SCHEMA_VERSION
from threatlens.utils.logger import get_logger

logger = get_logger("pe_extractor")

# Set of standard suspicious API calls monitored during static import analysis
SUSPICIOUS_APIS: Set[str] = {
    "virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex",
    "writeprocessmemory", "readprocessmemory", "createremotethread",
    "loadlibrarya", "loadlibraryw", "loadlibraryexa", "loadlibraryexw",
    "getprocaddress", "createprocessa", "createprocessw", "winexec",
    "isdebuggerpresent", "checkremotedebuggerpresent", "ntqueryinformationprocess",
    "urldownloadtofilea", "urldownloadtofilew", "internetopenurla", "internetopenurlw",
    "setwindowshookexa", "setwindowshookexw", "adjusttokenprivileges",
}


class PEParseError(Exception):
    """Custom exception raised when PE parsing or feature extraction fails."""
    pass


def calculate_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of a byte buffer.

    Returns:
        Float value between 0.0 and 8.0.
    """
    if not data:
        return 0.0

    length = len(data)
    byte_counts = [0] * 256
    for b in data:
        byte_counts[b] += 1

    entropy = 0.0
    for count in byte_counts:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return float(entropy)


class PEFeatureExtractor:
    """
    Static feature extractor for Windows PE32 and PE32+ executables.

    Extracts:
    - File metadata (size, machine, compilation timestamp, checksum)
    - PE Headers (DOS header, COFF file header, Optional header, flags)
    - Section characteristics (sizes, entropy, permissions)
    - Import tables (DLLs, functions, anomalous indicators)
    - Export tables (counts, function names, RVAs)
    - Resource directories (types, languages, sizes)
    - Additional attributes (overlay, digital certificate, debug data)

    Never executes the target binary.
    """

    def __init__(
        self,
        pefile_path: Optional[Union[str, Path]] = None,
        data: Optional[bytes] = None,
    ):
        """
        Initialize the extractor with either a file path or raw bytes.

        Args:
            pefile_path: Path to the PE file on disk.
            data: Raw binary byte content of a PE file.
        """
        self.pefile_path: Optional[Path] = Path(pefile_path) if pefile_path is not None else None
        self._raw_data: Optional[bytes] = data
        self.pe: Optional[pefile.PE] = None
        self.features: Dict[str, Any] = {}
        self._file_size: int = 0

        self.load_pefile()

    def load_pefile(self) -> None:
        """
        Load and parse the PE binary into memory with robust exception wrapping.
        """
        if self._raw_data is not None:
            self._file_size = len(self._raw_data)
            if self._file_size == 0:
                raise PEParseError("Cannot parse empty PE byte buffer (0 bytes)")
            try:
                self.pe = pefile.PE(data=self._raw_data, fast_load=False)
            except pefile.PEFormatError as e:
                raise PEParseError(f"Invalid PE format: {e}") from e
            except Exception as e:
                raise PEParseError(f"Unexpected error loading PE from bytes: {e}") from e
        elif self.pefile_path is not None:
            if not self.pefile_path.exists():
                raise PEParseError(f"File not found: {self.pefile_path}")
            if not self.pefile_path.is_file():
                raise PEParseError(f"Path is not a regular file: {self.pefile_path}")

            self._file_size = self.pefile_path.stat().st_size
            if self._file_size == 0:
                raise PEParseError(f"Empty file (0 bytes): {self.pefile_path}")

            try:
                # Read entire file safely
                with open(self.pefile_path, "rb") as f:
                    file_bytes = f.read()
                self._raw_data = file_bytes
                self.pe = pefile.PE(data=file_bytes, fast_load=False)
            except pefile.PEFormatError as e:
                logger.warning(f"PEFormatError for {self.pefile_path}: {e}")
                raise PEParseError(f"Invalid PE format: {e}") from e
            except Exception as e:
                logger.warning(f"Error parsing PE file {self.pefile_path}: {e}")
                raise PEParseError(f"Failed to parse PE file: {e}") from e
        else:
            raise PEParseError("Neither pefile_path nor data was provided to PEFeatureExtractor")

        # Validate core headers
        if not hasattr(self.pe, "DOS_HEADER") or self.pe.DOS_HEADER is None:
            raise PEParseError("PE file is missing DOS_HEADER")
        if getattr(self.pe.DOS_HEADER, "e_magic", 0) != 0x5A4D:
            raise PEParseError(f"Invalid DOS header magic: {hex(getattr(self.pe.DOS_HEADER, 'e_magic', 0))}")
        if not hasattr(self.pe, "FILE_HEADER") or self.pe.FILE_HEADER is None:
            raise PEParseError("PE file is missing FILE_HEADER")
        if not hasattr(self.pe, "OPTIONAL_HEADER") or self.pe.OPTIONAL_HEADER is None:
            raise PEParseError("PE file is missing OPTIONAL_HEADER")

    def extract(self) -> Dict[str, Any]:
        """
        Extract all static structural features into a dictionary matching PE_FEATURE_SCHEMA.

        Returns:
            Dictionary containing all extracted features.
        """
        if not self.pe:
            raise PEParseError("PE file not loaded or invalid.")

        self.features = {
            "schema_version": SCHEMA_VERSION,
        }

        try:
            self._extract_file_info()
            self._extract_dos_header()
            self._extract_file_header()
            self._extract_optional_header()
            self._extract_sections()
            self._extract_imports()
            self._extract_exports()
            self._extract_resources()
            self._extract_additional_characteristics()
        except PEParseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during static feature extraction: {e}")
            raise PEParseError(f"Unexpected extraction error: {e}") from e

        return self.features

    def _extract_file_info(self) -> None:
        """Extract general file metadata (size, machine, timestamp, checksum)."""
        self.features["file_size"] = self._file_size

        opt_header = self.pe.OPTIONAL_HEADER
        magic = getattr(opt_header, "Magic", 0)
        if magic == 0x10B:
            file_type = "PE32"
        elif magic == 0x20B:
            file_type = "PE32+"
        else:
            file_type = "UNKNOWN"
        self.features["file_type"] = file_type

        file_header = self.pe.FILE_HEADER
        machine_arch = getattr(file_header, "Machine", 0)
        self.features["machine_architecture"] = machine_arch

        # Machine name lookup
        machine_name = pefile.MACHINE_TYPE.get(machine_arch, f"IMAGE_FILE_MACHINE_UNKNOWN_{hex(machine_arch)}")
        if isinstance(machine_name, tuple):
            machine_name = machine_name[0]
        self.features["machine_name"] = str(machine_name)

        # Timestamps
        raw_timestamp = getattr(file_header, "TimeDateStamp", 0)
        self.features["pe_timestamp"] = raw_timestamp
        try:
            iso_ts = datetime.fromtimestamp(raw_timestamp, tz=timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            iso_ts = "1970-01-01T00:00:00+00:00"
        self.features["pe_timestamp_iso"] = iso_ts

        # Checksum
        self.features["checksum"] = getattr(opt_header, "CheckSum", 0)

    def _extract_dos_header(self) -> None:
        """Extract DOS header characteristics."""
        dos = self.pe.DOS_HEADER
        e_magic = getattr(dos, "e_magic", 0)
        e_lfanew = getattr(dos, "e_lfanew", 0)

        self.features["dos_header_magic"] = e_magic
        self.features["dos_header_pe_link"] = e_lfanew

        self.features["dos_header"] = {
            "e_magic": e_magic,
            "e_cblp": getattr(dos, "e_cblp", 0),
            "e_cp": getattr(dos, "e_cp", 0),
            "e_crlc": getattr(dos, "e_crlc", 0),
            "e_cparhdr": getattr(dos, "e_cparhdr", 0),
            "e_minalloc": getattr(dos, "e_minalloc", 0),
            "e_maxalloc": getattr(dos, "e_maxalloc", 0),
            "e_ss": getattr(dos, "e_ss", 0),
            "e_sp": getattr(dos, "e_sp", 0),
            "e_csum": getattr(dos, "e_csum", 0),
            "e_ip": getattr(dos, "e_ip", 0),
            "e_cs": getattr(dos, "e_cs", 0),
            "e_lfarlc": getattr(dos, "e_lfarlc", 0),
            "e_ovno": getattr(dos, "e_ovno", 0),
            "e_lfanew": e_lfanew,
        }

    def _extract_file_header(self) -> None:
        """Extract COFF file header characteristics and flags."""
        fh = self.pe.FILE_HEADER
        chars = getattr(fh, "Characteristics", 0)

        self.features["file_header_characteristics"] = chars
        self.features["file_header_number_of_sections"] = getattr(fh, "NumberOfSections", 0)
        self.features["file_header_pointer_symbol_table"] = getattr(fh, "PointerToSymbolTable", 0)
        self.features["file_header_number_of_symbols"] = getattr(fh, "NumberOfSymbols", 0)
        self.features["file_header_size_optional_header"] = getattr(fh, "SizeOfOptionalHeader", 0)

        # Characteristics flags
        self.features["characteristics"] = {
            "relocs_stripped": bool(chars & 0x0001),
            "executable_image": bool(chars & 0x0002),
            "line_nums_stripped": bool(chars & 0x0004),
            "local_syms_stripped": bool(chars & 0x0008),
            "large_address_aware": bool(chars & 0x0020),
            "32bit_machine": bool(chars & 0x0100),
            "debug_stripped": bool(chars & 0x0200),
            "removable_run_from_swap": bool(chars & 0x0400),
            "net_run_from_swap": bool(chars & 0x0800),
            "system": bool(chars & 0x1000),
            "dll": bool(chars & 0x2000),
            "up_system_only": bool(chars & 0x4000),
        }

    def _extract_optional_header(self) -> None:
        """Extract Optional header characteristics and DLL characteristics."""
        opt = self.pe.OPTIONAL_HEADER

        self.features["entry_point"] = getattr(opt, "AddressOfEntryPoint", 0)
        self.features["image_base"] = getattr(opt, "ImageBase", 0)
        subsystem_id = getattr(opt, "Subsystem", 0)
        self.features["subsystem"] = subsystem_id

        # Subsystem name lookup
        subsystem_name = pefile.SUBSYSTEM_TYPE.get(subsystem_id, f"IMAGE_SUBSYSTEM_UNKNOWN_{subsystem_id}")
        self.features["subsystem_name"] = str(subsystem_name)

        self.features["optional_header_magic"] = getattr(opt, "Magic", 0)
        self.features["optional_header_major_linker_version"] = getattr(opt, "MajorLinkerVersion", 0)
        self.features["optional_header_minor_linker_version"] = getattr(opt, "MinorLinkerVersion", 0)
        self.features["optional_header_size_of_code"] = getattr(opt, "SizeOfCode", 0)
        self.features["optional_header_size_of_initialized_data"] = getattr(opt, "SizeOfInitializedData", 0)
        self.features["optional_header_size_of_uninitialized_data"] = getattr(opt, "SizeOfUninitializedData", 0)
        self.features["optional_header_base_of_code"] = getattr(opt, "BaseOfCode", 0)

        # BaseOfData only exists in PE32 (32-bit); absent in PE32+ (64-bit)
        self.features["optional_header_base_of_data"] = getattr(opt, "BaseOfData", None)

        self.features["optional_header_image_size"] = getattr(opt, "SizeOfImage", 0)
        self.features["optional_header_size_of_headers"] = getattr(opt, "SizeOfHeaders", 0)
        self.features["optional_header_checksum"] = getattr(opt, "CheckSum", 0)

        dll_chars = getattr(opt, "DllCharacteristics", 0)
        self.features["optional_header_dll_characteristics"] = dll_chars

        self.features["optional_header_size_of_stack_reserve"] = getattr(opt, "SizeOfStackReserve", 0)
        self.features["optional_header_size_of_stack_commit"] = getattr(opt, "SizeOfStackCommit", 0)
        self.features["optional_header_size_of_heap_reserve"] = getattr(opt, "SizeOfHeapReserve", 0)
        self.features["optional_header_size_of_heap_commit"] = getattr(opt, "SizeOfHeapCommit", 0)
        self.features["optional_header_number_of_rva_and_sizes"] = getattr(opt, "NumberOfRvaAndSizes", 0)
        self.features["optional_header_section_alignment"] = getattr(opt, "SectionAlignment", 0)
        self.features["optional_header_file_alignment"] = getattr(opt, "FileAlignment", 0)

        # DLL Characteristics flags
        self.features["dll_characteristics_flags"] = {
            "high_entropy_va": bool(dll_chars & 0x0020),
            "dynamic_base": bool(dll_chars & 0x0040),
            "force_integrity": bool(dll_chars & 0x0080),
            "nx_compat": bool(dll_chars & 0x0100),
            "no_isolation": bool(dll_chars & 0x0200),
            "no_seh": bool(dll_chars & 0x0400),
            "no_bind": bool(dll_chars & 0x0800),
            "appcontainer": bool(dll_chars & 0x1000),
            "wdm_driver": bool(dll_chars & 0x2000),
            "guard_cf": bool(dll_chars & 0x4000),
            "terminal_server_aware": bool(dll_chars & 0x8000),
        }

    def _get_section_permissions(self, chars: int) -> Dict[str, bool]:
        """Interprets section characteristics bitmask into readable permissions."""
        is_read = bool(chars & 0x40000000)
        is_write = bool(chars & 0x80000000)
        is_exec = bool(chars & 0x20000000)
        is_shared = bool(chars & 0x10000000)
        cnt_code = bool(chars & 0x00000020)
        cnt_init = bool(chars & 0x00000040)
        cnt_uninit = bool(chars & 0x00000080)

        # Default readable for sections containing code or initialized data
        if not is_read and (cnt_code or cnt_init):
            is_read = True

        return {
            "readable": is_read,
            "writable": is_write,
            "executable": is_exec,
            "shared": is_shared,
            "contains_code": cnt_code,
            "contains_initialized_data": cnt_init,
            "contains_uninitialized_data": cnt_uninit,
        }

    def _extract_sections(self) -> None:
        """Extracts detailed information and entropy for each section."""
        sections_data: List[Dict[str, Any]] = []

        if hasattr(self.pe, "sections") and self.pe.sections:
            for idx, sec in enumerate(self.pe.sections):
                raw_name = getattr(sec, "Name", b"")
                if isinstance(raw_name, bytes):
                    clean_name = raw_name.decode("utf-8", "ignore").rstrip("\x00").strip()
                else:
                    clean_name = str(raw_name).rstrip("\x00").strip()

                try:
                    entropy = float(sec.get_entropy())
                except Exception:
                    entropy = 0.0

                chars = getattr(sec, "Characteristics", 0)
                sec_dict = {
                    "index": idx,
                    "name": clean_name,
                    "virtual_address": getattr(sec, "VirtualAddress", 0),
                    "virtual_size": getattr(sec, "Misc_VirtualSize", 0),
                    "size_of_raw_data": getattr(sec, "SizeOfRawData", 0),
                    "pointer_to_raw_data": getattr(sec, "PointerToRawData", 0),
                    "entropy": entropy,
                    "characteristics": chars,
                    "permissions": self._get_section_permissions(chars),
                }
                sections_data.append(sec_dict)

        self.features["sections"] = sections_data
        self.features["number_of_sections"] = len(sections_data)

    def _extract_imports(self) -> None:
        """Extracts imported libraries, functions, and import anomalies."""
        dll_names: List[str] = []
        imports_data: List[Dict[str, Any]] = []
        total_functions = 0
        has_empty_dll = False
        has_ordinal_only = False
        suspicious_found: List[str] = []

        if hasattr(self.pe, "DIRECTORY_ENTRY_IMPORT") and self.pe.DIRECTORY_ENTRY_IMPORT:
            for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
                raw_dll = getattr(entry, "dll", b"")
                if isinstance(raw_dll, bytes):
                    dll_name = raw_dll.decode("utf-8", "ignore").strip()
                else:
                    dll_name = str(raw_dll).strip()

                if not dll_name:
                    has_empty_dll = True

                dll_names.append(dll_name)
                func_list: List[Dict[str, Any]] = []

                if hasattr(entry, "imports") and entry.imports:
                    for imp in entry.imports:
                        func_name_raw = getattr(imp, "name", None)
                        func_name: Optional[str] = None
                        if func_name_raw is not None:
                            if isinstance(func_name_raw, bytes):
                                func_name = func_name_raw.decode("utf-8", "ignore").strip()
                            else:
                                func_name = str(func_name_raw).strip()

                        ordinal = getattr(imp, "ordinal", None)
                        if func_name is None and ordinal is not None:
                            has_ordinal_only = True

                        addr = getattr(imp, "address", 0)
                        addr_str = hex(addr) if addr is not None else "0x0"

                        if func_name and func_name.lower() in SUSPICIOUS_APIS:
                            suspicious_found.append(func_name)

                        func_list.append({
                            "name": func_name,
                            "ordinal": ordinal,
                            "address": addr_str,
                        })
                        total_functions += 1

                imports_data.append({
                    "dll_name": dll_name,
                    "function_count": len(func_list),
                    "imported_functions": func_list,
                })

        self.features["number_of_imported_dlls"] = len(imports_data)
        self.features["number_of_imported_functions"] = total_functions
        self.features["imported_dll_names"] = dll_names
        self.features["imports"] = imports_data
        self.features["unusual_import_characteristics"] = {
            "has_empty_dll_names": has_empty_dll,
            "has_ordinal_only_imports": has_ordinal_only,
            "suspicious_import_count": len(suspicious_found),
            "suspicious_imports_found": sorted(list(set(suspicious_found))),
        }

    def _extract_exports(self) -> None:
        """Extracts exported functions from DIRECTORY_ENTRY_EXPORT."""
        export_names: List[str] = []
        exports_data: List[Dict[str, Any]] = []

        if hasattr(self.pe, "DIRECTORY_ENTRY_EXPORT") and self.pe.DIRECTORY_ENTRY_EXPORT:
            exp_dir = self.pe.DIRECTORY_ENTRY_EXPORT
            symbols = getattr(exp_dir, "symbols", [])
            for sym in symbols:
                sym_name_raw = getattr(sym, "name", None)
                if sym_name_raw is not None:
                    if isinstance(sym_name_raw, bytes):
                        sym_name = sym_name_raw.decode("utf-8", "ignore").strip()
                    else:
                        sym_name = str(sym_name_raw).strip()
                    export_names.append(sym_name)
                else:
                    sym_name = None

                ordinal = getattr(sym, "ordinal", 0)
                addr = getattr(sym, "address", 0)
                exports_data.append({
                    "name": sym_name,
                    "ordinal": ordinal,
                    "address": hex(addr) if addr is not None else "0x0",
                })

        self.features["number_of_exports"] = len(exports_data)
        self.features["export_names"] = export_names
        self.features["exports"] = exports_data

    def _extract_resources(self) -> None:
        """Extracts resource hierarchy (types, IDs, languages, sizes)."""
        resources_list: List[Dict[str, Any]] = []
        resource_types: Set[str] = set()

        if hasattr(self.pe, "DIRECTORY_ENTRY_RESOURCE") and self.pe.DIRECTORY_ENTRY_RESOURCE:
            try:
                for res_type in getattr(self.pe.DIRECTORY_ENTRY_RESOURCE, "entries", []):
                    # Type name or ID
                    if getattr(res_type, "name", None):
                        type_name = str(res_type.name)
                    else:
                        type_id = getattr(res_type, "id", 0)
                        type_name = pefile.RESOURCE_TYPE.get(type_id, f"RT_UNKNOWN_{type_id}")
                    resource_types.add(str(type_name))

                    if not hasattr(res_type, "directory") or not res_type.directory:
                        continue

                    for res_id in getattr(res_type.directory, "entries", []):
                        item_id = getattr(res_id, "id", 0)

                        if hasattr(res_id, "directory") and res_id.directory:
                            for lang_entry in getattr(res_id.directory, "entries", []):
                                lang_id = getattr(lang_entry, "id", 0)
                                lang_name = pefile.LANG.get(lang_id & 0x3FF, str(lang_id))

                                size = 0
                                if hasattr(lang_entry, "data") and hasattr(lang_entry.data, "struct"):
                                    size = getattr(lang_entry.data.struct, "Size", 0)

                                resources_list.append({
                                    "type_name": str(type_name),
                                    "id": item_id if isinstance(item_id, int) else 0,
                                    "lang": str(lang_name),
                                    "size": size,
                                })
                        elif hasattr(res_id, "data") and hasattr(res_id.data, "struct"):
                            size = getattr(res_id.data.struct, "Size", 0)
                            resources_list.append({
                                "type_name": str(type_name),
                                "id": item_id if isinstance(item_id, int) else 0,
                                "lang": "UNKNOWN",
                                "size": size,
                            })
            except Exception as e:
                logger.warning(f"Error parsing resource tree: {e}")

        self.features["has_resources"] = len(resources_list) > 0
        self.features["resource_count"] = len(resources_list)
        self.features["resource_types"] = sorted(list(resource_types))
        self.features["resources"] = resources_list

    def _extract_additional_characteristics(self) -> None:
        """Extracts overlay data, digital certificate table, and debug directory info."""
        # Overlay extraction
        overlay_bytes: Optional[bytes] = None
        try:
            overlay_bytes = self.pe.get_overlay()
        except Exception as e:
            logger.debug(f"Could not retrieve overlay: {e}")

        if overlay_bytes:
            overlay_size = len(overlay_bytes)
            overlay_entropy = calculate_entropy(overlay_bytes)
            has_overlay = True
        else:
            overlay_size = 0
            overlay_entropy = 0.0
            has_overlay = False

        self.features["overlay_size"] = overlay_size
        self.features["has_overlay"] = has_overlay
        self.features["overlay_entropy"] = overlay_entropy

        # Certificate / Security directory
        cert_present = False
        cert_va = 0
        cert_size = 0
        try:
            sec_idx = pefile.DIRECTORY_ENTRY.get("IMAGE_DIRECTORY_ENTRY_SECURITY", 4)
            if hasattr(self.pe, "OPTIONAL_HEADER") and hasattr(self.pe.OPTIONAL_HEADER, "DATA_DIRECTORY"):
                data_dirs = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY
                if len(data_dirs) > sec_idx:
                    sec_dir = data_dirs[sec_idx]
                    cert_va = getattr(sec_dir, "VirtualAddress", 0)
                    cert_size = getattr(sec_dir, "Size", 0)
                    cert_present = (cert_size > 0 and cert_va > 0)
        except Exception as e:
            logger.debug(f"Could not check security certificate directory: {e}")

        self.features["certificate_table"] = {
            "present": cert_present,
            "virtual_address": cert_va,
            "size": cert_size,
        }

        # Debug directory
        debug_count = 0
        debug_types: List[str] = []
        has_debug = False

        if hasattr(self.pe, "DIRECTORY_ENTRY_DEBUG") and self.pe.DIRECTORY_ENTRY_DEBUG:
            debug_entries = self.pe.DIRECTORY_ENTRY_DEBUG
            debug_count = len(debug_entries)
            has_debug = debug_count > 0
            for dbg in debug_entries:
                dbg_type_id = getattr(getattr(dbg, "struct", None), "Type", 0)
                dbg_type_name = pefile.DEBUG_TYPE.get(dbg_type_id, f"IMAGE_DEBUG_TYPE_{dbg_type_id}")
                debug_types.append(str(dbg_type_name))

        self.features["debug_entries_count"] = debug_count
        self.features["has_debug_data"] = has_debug
        self.features["debug_types"] = debug_types


def extract_pe_features(
    file_path: Optional[Union[str, Path]] = None,
    data: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Convenience function to extract PE static features.

    Args:
        file_path: Path to PE executable on disk.
        data: In-memory raw bytes of a PE executable.

    Returns:
        Extracted features dictionary.
    """
    extractor = PEFeatureExtractor(pefile_path=file_path, data=data)
    return extractor.extract()


__all__ = [
    "PEFeatureExtractor",
    "PEParseError",
    "calculate_entropy",
    "extract_pe_features",
]
