"""
EMBER 2018 Dataset Loader

Handles downloading, extracting, and memory-efficient streaming/chunked loading of the
genuine EMBER 2018 dataset (900,000 train + 200,000 test PE feature records).
Extracts the official 2,351-dimensional EMBER feature representation (Feature Version 1).
"""

from dataclasses import dataclass, field
import glob
import json
import logging
import os
from pathlib import Path
import tarfile
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher
from tqdm import tqdm

from threatlens.config import DATA_CONFIG, FEATURE_GROUPS, EmberFeatureGroups
from threatlens.utils.logger import get_logger

logger = get_logger("ember_loader")


class EmberDatasetError(Exception):
    """Custom exception for EMBER dataset operations."""
    pass


@dataclass
class LoaderStats:
    """Provenance and count statistics for loaded EMBER records."""
    total_records_read: int = 0
    unlabeled_skipped: int = 0
    malformed_skipped: int = 0
    benign_count: int = 0
    malicious_count: int = 0
    unique_sha256_count: int = 0
    files_processed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records_read": self.total_records_read,
            "unlabeled_skipped": self.unlabeled_skipped,
            "malformed_skipped": self.malformed_skipped,
            "benign_count": self.benign_count,
            "malicious_count": self.malicious_count,
            "unique_sha256_count": self.unique_sha256_count,
            "files_processed": list(self.files_processed),
        }


def iter_ember_jsonl(
    file_path: Union[str, Path],
    skip_unlabeled: bool = True,
    seen_sha256: Optional[Set[str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream individual JSON records from an EMBER JSONL file safely.
    Skips empty lines, whitespace, malformed JSON, and optionally unlabeled (-1) samples.
    """
    p = Path(file_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = rec.get("label", -1)
            if skip_unlabeled and label == -1:
                continue
            sha256 = rec.get("sha256", "")
            if seen_sha256 and sha256 in seen_sha256:
                continue
            yield rec


def vectorize_ember_batch(records: List[dict]) -> np.ndarray:
    """
    Vectorize a batch of raw EMBER 2018 JSON records into a 2D numpy array of shape (N, 2351).
    Matches the official EMBER Feature Version 1 specification exactly:
      1. ByteHistogram: 256
      2. ByteEntropyHistogram: 256
      3. StringExtractor: 104
      4. GeneralFileInfo: 10
      5. HeaderFileInfo: 62
      6. SectionInfo: 255
      7. ImportsInfo: 1280
      8. ExportsInfo: 128
      Total = 2,351 dimensions.
    """
    n = len(records)
    if n == 0:
        return np.zeros((0, EmberFeatureGroups.TOTAL_FEATURES), dtype=np.float32)

    # 1. ByteHistogram (256)
    hists = np.array([r.get("histogram", [0] * 256) for r in records], dtype=np.float32)
    if hists.shape[1] != 256:
        hists = np.pad(hists, ((0, 0), (0, max(0, 256 - hists.shape[1]))))[:, :256]
    h_sums = hists.sum(axis=1, keepdims=True)
    h_sums[h_sums == 0] = 1.0
    hists /= h_sums

    # 2. ByteEntropyHistogram (256)
    entropies = np.array([r.get("byteentropy", [0] * 256) for r in records], dtype=np.float32)
    if entropies.shape[1] != 256:
        entropies = np.pad(entropies, ((0, 0), (0, max(0, 256 - entropies.shape[1]))))[:, :256]
    e_sums = entropies.sum(axis=1, keepdims=True)
    e_sums[e_sums == 0] = 1.0
    entropies /= e_sums

    # 3. StringExtractor (104)
    # [numstrings, avlength, printables, printabledist (96), entropy, paths, urls, registry, MZ]
    s_scalars_list = []
    s_dists_list = []
    for r in records:
        st = r.get("strings", {})
        p_count = float(st.get("printables", 0))
        divisor = p_count if p_count > 0 else 1.0
        p_dist = st.get("printabledist", [0] * 96)
        if len(p_dist) < 96:
            p_dist = list(p_dist) + [0] * (96 - len(p_dist))
        dist_norm = (np.asarray(p_dist[:96], dtype=np.float32) / divisor)
        s_scalars_list.append([
            st.get("numstrings", 0),
            st.get("avlength", 0),
            st.get("printables", 0),
            st.get("entropy", 0),
            st.get("paths", 0),
            st.get("urls", 0),
            st.get("registry", 0),
            st.get("MZ", 0),
        ])
        s_dists_list.append(dist_norm)

    s_scalars = np.array(s_scalars_list, dtype=np.float32)
    s_dists = np.array(s_dists_list, dtype=np.float32)
    strings_mat = np.hstack([s_scalars[:, :3], s_dists, s_scalars[:, 3:]]).astype(np.float32)

    # 4. GeneralFileInfo (10)
    gen_list = []
    for r in records:
        g = r.get("general", {})
        gen_list.append([
            g.get("size", 0),
            g.get("vsize", 0),
            g.get("has_debug", 0),
            g.get("exports", 0),
            g.get("imports", 0),
            g.get("has_relocations", 0),
            g.get("has_resources", 0),
            g.get("has_signature", 0),
            g.get("has_tls", 0),
            g.get("symbols", 0),
        ])
    gen_mat = np.array(gen_list, dtype=np.float32)

    # 5. HeaderFileInfo (62)
    # [timestamp, machine 10, chars 10, subsystem 10, dll_chars 10, magic 10, + 11 scalar fields]
    timestamps = []
    coff_mach_list = []
    coff_char_list = []
    opt_sub_list = []
    opt_dll_list = []
    opt_mag_list = []
    h_scalars_list = []

    for r in records:
        h = r.get("header", {})
        coff = h.get("coff", {})
        opt = h.get("optional", {})
        timestamps.append([coff.get("timestamp", 0)])
        coff_mach_list.append([str(coff.get("machine", ""))])
        coff_char_list.append([str(c) for c in coff.get("characteristics", [])])
        opt_sub_list.append([str(opt.get("subsystem", ""))])
        opt_dll_list.append([str(c) for c in opt.get("dll_characteristics", [])])
        opt_mag_list.append([str(opt.get("magic", ""))])
        h_scalars_list.append([
            opt.get("major_image_version", 0),
            opt.get("minor_image_version", 0),
            opt.get("major_linker_version", 0),
            opt.get("minor_linker_version", 0),
            opt.get("major_operating_system_version", 0),
            opt.get("minor_operating_system_version", 0),
            opt.get("major_subsystem_version", 0),
            opt.get("minor_subsystem_version", 0),
            opt.get("sizeof_code", 0),
            opt.get("sizeof_headers", 0),
            opt.get("sizeof_heap_commit", 0),
        ])

    ts_mat = np.array(timestamps, dtype=np.float32)
    h_mach = FeatureHasher(10, input_type="string").transform(coff_mach_list).toarray().astype(np.float32)
    h_char = FeatureHasher(10, input_type="string").transform(coff_char_list).toarray().astype(np.float32)
    h_sub = FeatureHasher(10, input_type="string").transform(opt_sub_list).toarray().astype(np.float32)
    h_dll = FeatureHasher(10, input_type="string").transform(opt_dll_list).toarray().astype(np.float32)
    h_mag = FeatureHasher(10, input_type="string").transform(opt_mag_list).toarray().astype(np.float32)
    h_scalars = np.array(h_scalars_list, dtype=np.float32)
    header_mat = np.hstack([ts_mat, h_mach, h_char, h_sub, h_dll, h_mag, h_scalars])

    # 6. SectionInfo (255)
    sec_generals = []
    sec_sizes_all = []
    sec_entropy_all = []
    sec_vsize_all = []
    sec_entry_all = []
    sec_chars_all = []

    for r in records:
        sec_obj = r.get("section", {})
        secs = sec_obj.get("sections", []) if isinstance(sec_obj, dict) else []
        if not isinstance(secs, list):
            secs = []
        entry = str(sec_obj.get("entry", "")) if isinstance(sec_obj, dict) else ""

        g = [
            len(secs),
            sum(1 for s in secs if isinstance(s, dict) and s.get("size", 0) == 0),
            sum(1 for s in secs if isinstance(s, dict) and s.get("name", "") == ""),
            sum(1 for s in secs if isinstance(s, dict) and "MEM_READ" in s.get("props", []) and "MEM_EXECUTE" in s.get("props", [])),
            sum(1 for s in secs if isinstance(s, dict) and "MEM_WRITE" in s.get("props", [])),
        ]
        sec_generals.append(g)
        sec_sizes_all.append([(str(s.get("name", "")), float(s.get("size", 0))) for s in secs if isinstance(s, dict)])
        sec_entropy_all.append([(str(s.get("name", "")), float(s.get("entropy", 0))) for s in secs if isinstance(s, dict)])
        sec_vsize_all.append([(str(s.get("name", "")), float(s.get("vsize", s.get("virtual_size", 0)))) for s in secs if isinstance(s, dict)])
        sec_entry_all.append([entry])
        sec_chars_all.append([str(p) for s in secs if isinstance(s, dict) and s.get("name", "") == entry for p in s.get("props", [])])

    sec_g_mat = np.array(sec_generals, dtype=np.float32)
    sec_sizes_h = FeatureHasher(50, input_type="pair").transform(sec_sizes_all).toarray().astype(np.float32)
    sec_entropy_h = FeatureHasher(50, input_type="pair").transform(sec_entropy_all).toarray().astype(np.float32)
    sec_vsize_h = FeatureHasher(50, input_type="pair").transform(sec_vsize_all).toarray().astype(np.float32)
    sec_entry_h = FeatureHasher(50, input_type="string").transform(sec_entry_all).toarray().astype(np.float32)
    sec_chars_h = FeatureHasher(50, input_type="string").transform(sec_chars_all).toarray().astype(np.float32)
    sec_mat = np.hstack([sec_g_mat, sec_sizes_h, sec_entropy_h, sec_vsize_h, sec_entry_h, sec_chars_h])

    # 7. ImportsInfo (1280)
    imp_libs_all = []
    imp_funcs_all = []
    for r in records:
        imp_obj = r.get("imports", {})
        if isinstance(imp_obj, dict):
            libs = list(set([str(l).lower() for l in imp_obj.keys()]))
            funcs = [
                str(lib).lower() + ":" + str(e)
                for lib, elist in imp_obj.items()
                if isinstance(elist, list)
                for e in elist
            ]
        else:
            libs, funcs = [], []
        imp_libs_all.append(libs)
        imp_funcs_all.append(funcs)

    imp_libs_h = FeatureHasher(256, input_type="string").transform(imp_libs_all).toarray().astype(np.float32)
    imp_funcs_h = FeatureHasher(1024, input_type="string").transform(imp_funcs_all).toarray().astype(np.float32)
    imp_mat = np.hstack([imp_libs_h, imp_funcs_h])

    # 8. ExportsInfo (128)
    exp_all = []
    for r in records:
        exp_obj = r.get("exports", [])
        if isinstance(exp_obj, list):
            exp_all.append([str(e) for e in exp_obj])
        elif isinstance(exp_obj, dict):
            exp_all.append([str(e) for e in exp_obj.get("exports", [])])
        else:
            exp_all.append([])
    exp_mat = FeatureHasher(128, input_type="string").transform(exp_all).toarray().astype(np.float32)

    # Full 2,351-dimensional concatenation
    feature_matrix = np.hstack([
        hists, entropies, strings_mat, gen_mat, header_mat, sec_mat, imp_mat, exp_mat
    ]).astype(np.float32)

    # Replace any potential non-finite values (NaN / Inf) with 0.0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_matrix


def vectorize_ember_record(raw_record: dict) -> np.ndarray:
    """Vectorize a single raw EMBER 2018 JSON record to a 1D float32 array of shape (2351,)."""
    return vectorize_ember_batch([raw_record])[0]


class EmberLoader:
    """
    Streaming and chunk-based loader for the genuine EMBER 2018 dataset.
    Supports memory-efficient loading across all six training files and official test isolation.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the EMBER loader.

        Args:
            data_dir: Base directory containing EMBER dataset files
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir or DATA_CONFIG.RAW_DATA_DIR)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.stats = LoaderStats()
        self._seen_train_sha256: Set[str] = set()

    def get_effective_raw_dir(self) -> Path:
        """Locate the directory containing extracted EMBER files."""
        ember_subdir = self.data_dir / "ember2018"
        if ember_subdir.exists():
            return ember_subdir
        return self.data_dir

    def _get_train_file_paths(self) -> List[Path]:
        """Return list of existing EMBER training JSONL files."""
        raw_dir = self.get_effective_raw_dir()
        paths = []
        for i in range(6):
            fpath = raw_dir / f"train_features_{i}.jsonl"
            if fpath.exists():
                paths.append(fpath)

        # Fallback to single train_features.jsonl if present
        single_path = raw_dir / DATA_CONFIG.EMBER_TRAIN_FEATURES
        if not paths and single_path.exists():
            paths.append(single_path)

        return paths

    def _get_test_file_path(self) -> Optional[Path]:
        """Return path to official test_features.jsonl."""
        raw_dir = self.get_effective_raw_dir()
        test_path = raw_dir / DATA_CONFIG.EMBER_TEST_FEATURES
        if test_path.exists():
            return test_path
        return None

    def _check_files_exist(self) -> bool:
        """Check if genuine EMBER training files exist."""
        train_paths = self._get_train_file_paths()
        test_path = self._get_test_file_path()
        return len(train_paths) > 0 and test_path is not None and test_path.exists()

    def download_dataset(self, force: bool = False) -> bool:
        """Download EMBER 2018 dataset from official source."""
        archive_path = self.data_dir / DATA_CONFIG.EMBER_ARCHIVE_NAME

        if not force and self._check_files_exist():
            logger.info("EMBER dataset files already exist, skipping download")
            return True

        logger.info(f"Downloading EMBER 2018 dataset from {DATA_CONFIG.EMBER_URL}")
        try:
            response = requests.get(DATA_CONFIG.EMBER_URL, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            self.data_dir.mkdir(parents=True, exist_ok=True)

            with open(archive_path, "wb") as f, tqdm(
                desc="Downloading EMBER",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info("Download complete. Extracting archive...")
            self._extract_archive(archive_path)
            logger.info("EMBER dataset ready")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to download EMBER dataset: {e}")
            raise EmberDatasetError(f"Download failed: {e}") from e
        except tarfile.TarError as e:
            logger.error(f"Failed to extract EMBER archive: {e}")
            raise EmberDatasetError(f"Extraction failed: {e}") from e

    def _extract_archive(self, archive_path: Path) -> None:
        """Safely extract the EMBER tar.bz2 archive with path traversal checks."""
        with tarfile.open(archive_path, "r:bz2") as tar:
            target_dir = self.data_dir.resolve()
            for member in tar.getmembers():
                dest = (target_dir / member.name).resolve()
                if not str(dest).startswith(str(target_dir)):
                    raise EmberDatasetError(f"Path traversal detected in archive: {member.name}")
            tar.extractall(path=self.data_dir)

    def stream_records(
        self,
        file_paths: List[Path],
        include_unlabeled: bool = False,
        max_samples: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, List[str]], None, None]:
        """
        Stream records in vectorized batches from JSONL files without holding the entire dataset in RAM.

        Yields:
            Tuples of (batch_features_array (K, 2351), batch_labels_array (K,), batch_sha256_list)
        """
        collected_total = 0

        for fpath in file_paths:
            if not fpath.exists():
                logger.warning(f"File not found: {fpath}")
                continue

            self.stats.files_processed.append(fpath.name)
            logger.info(f"Streaming records from {fpath.name}...")

            batch_records: List[dict] = []
            batch_labels: List[int] = []
            batch_hashes: List[str] = []

            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    self.stats.total_records_read += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as err:
                        self.stats.malformed_skipped += 1
                        logger.debug(f"{fpath.name}:{line_idx} malformed JSON: {err}")
                        continue

                    raw_label = record.get("label", -1)
                    if raw_label == -1 and not include_unlabeled:
                        self.stats.unlabeled_skipped += 1
                        continue

                    label_int = int(raw_label)
                    sha256 = record.get("sha256", f"unknown_{self.stats.total_records_read}")

                    if label_int == 0:
                        self.stats.benign_count += 1
                    elif label_int == 1:
                        self.stats.malicious_count += 1

                    batch_records.append(record)
                    batch_labels.append(label_int)
                    batch_hashes.append(sha256)

                    if len(batch_records) >= chunk_size:
                        features = vectorize_ember_batch(batch_records)
                        labels = np.array(batch_labels, dtype=np.int32)
                        yield features, labels, batch_hashes

                        collected_total += len(batch_records)
                        batch_records = []
                        batch_labels = []
                        batch_hashes = []

                        if max_samples and collected_total >= max_samples:
                            break

                if batch_records and (not max_samples or collected_total < max_samples):
                    features = vectorize_ember_batch(batch_records)
                    labels = np.array(batch_labels, dtype=np.int32)
                    yield features, labels, batch_hashes
                    collected_total += len(batch_records)

            if max_samples and collected_total >= max_samples:
                break

    def load_train_data(
        self,
        max_samples: Optional[int] = None,
        include_unlabeled: bool = False,
        balance_classes: bool = True,
        chunk_size: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training features and labels from genuine EMBER 2018 JSONL training files.
        When max_samples is given, loads equal balanced numbers of benign (0) and malicious (1) samples.
        """
        train_paths = self._get_train_file_paths()
        if not train_paths:
            raise EmberDatasetError("No EMBER training files found.")

        target_benign = (max_samples // 2) if (max_samples and balance_classes) else None
        target_malicious = (max_samples - (max_samples // 2)) if (max_samples and balance_classes) else None
        benign_records: List[dict] = []
        malicious_records: List[dict] = []
        benign_hashes: List[str] = []
        malicious_hashes: List[str] = []

        logger.info(f"Loading genuine EMBER training data from {len(train_paths)} files...")

        for fpath in train_paths:
            is_benign_only = fpath.name.endswith("_0.jsonl")
            if is_benign_only and target_benign is not None and len(benign_records) >= target_benign:
                continue

            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.stats.total_records_read += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        self.stats.malformed_skipped += 1
                        continue

                    label = rec.get("label", -1)
                    if label == -1 and not include_unlabeled:
                        self.stats.unlabeled_skipped += 1
                        continue

                    sha256 = rec.get("sha256", "")
                    if label == 0:
                        if target_benign is None or len(benign_records) < target_benign:
                            benign_records.append(rec)
                            benign_hashes.append(sha256)
                            self.stats.benign_count += 1
                    elif label == 1:
                        if target_malicious is None or len(malicious_records) < target_malicious:
                            malicious_records.append(rec)
                            malicious_hashes.append(sha256)
                            self.stats.malicious_count += 1

                    if target_benign is not None:
                        if is_benign_only and len(benign_records) >= target_benign:
                            break
                        if len(benign_records) >= target_benign and len(malicious_records) >= target_malicious:
                            break

            if target_benign is not None and len(benign_records) >= target_benign and len(malicious_records) >= target_malicious:
                break

        # Record seen SHA-256 hashes to prevent test set contamination
        for h in benign_hashes + malicious_hashes:
            if h:
                self._seen_train_sha256.add(h)
        self.stats.unique_sha256_count = len(self._seen_train_sha256)

        # Combine and shuffle
        all_records = benign_records + malicious_records
        all_labels = [0] * len(benign_records) + [1] * len(malicious_records)

        perm = self.rng.permutation(len(all_records))
        shuffled_records = [all_records[i] for i in perm]
        y = np.array([all_labels[i] for i in perm], dtype=np.int32)

        # Batch vectorization
        logger.info(f"Vectorizing {len(shuffled_records)} training records (Benign: {len(benign_records)}, Malicious: {len(malicious_records)})...")
        X = vectorize_ember_batch(shuffled_records)

        logger.info(f"Vectorization complete: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def load_test_data(
        self,
        max_samples: Optional[int] = None,
        balance_classes: bool = True,
        chunk_size: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load isolated official test data strictly from test_features.jsonl.
        Guarantees zero samples overlap with seen training hashes.
        """
        test_path = self._get_test_file_path()
        if not test_path:
            raise EmberDatasetError("test_features.jsonl not found.")

        target_per_class = (max_samples // 2) if (max_samples and balance_classes) else None
        benign_records: List[dict] = []
        malicious_records: List[dict] = []
        skipped_leakage = 0

        logger.info(f"Loading official isolated test data from {test_path.name}...")

        with open(test_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = rec.get("label", -1)
                if label == -1:
                    continue

                sha256 = rec.get("sha256", "")
                if sha256 and sha256 in self._seen_train_sha256:
                    skipped_leakage += 1
                    continue

                if label == 0:
                    if target_per_class is None or len(benign_records) < target_per_class:
                        benign_records.append(rec)
                elif label == 1:
                    if target_per_class is None or len(malicious_records) < target_per_class:
                        malicious_records.append(rec)

                if target_per_class and len(benign_records) >= target_per_class and len(malicious_records) >= target_per_class:
                    break

        if skipped_leakage > 0:
            logger.warning(f"Prevented leakage: skipped {skipped_leakage} test samples present in training set.")

        all_records = benign_records + malicious_records
        all_labels = [0] * len(benign_records) + [1] * len(malicious_records)

        perm = self.rng.permutation(len(all_records))
        shuffled_records = [all_records[i] for i in perm]
        y = np.array([all_labels[i] for i in perm], dtype=np.int32)

        logger.info(f"Vectorizing {len(shuffled_records)} official test records (Benign: {len(benign_records)}, Malicious: {len(malicious_records)})...")
        X = vectorize_ember_batch(shuffled_records)

        return X, y

    def generate_synthetic_data(
        self,
        n_samples: Optional[int] = None,
        class_balance: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic EMBER-like data for testing when dataset is unavailable."""
        n_samples = n_samples or DATA_CONFIG.SYNTHETIC_SAMPLES
        n_features = EmberFeatureGroups.TOTAL_FEATURES

        logger.info(f"Generating {n_samples} synthetic samples with {n_features} features")
        features = np.zeros((n_samples, n_features), dtype=np.float32)

        for group_name in EmberFeatureGroups.get_all_group_names():
            start, end = EmberFeatureGroups.get_group_indices(group_name)
            group_size = end - start

            if group_name in ["histogram", "byteentropy"]:
                alpha = self.rng.uniform(0.1, 1.0, group_size)
                features[:, start:end] = self.rng.dirichlet(alpha, n_samples)
            elif group_name == "strings":
                sparse_mask = self.rng.random((n_samples, group_size)) < 0.05
                features[:, start:end] = sparse_mask * self.rng.exponential(1.0, (n_samples, group_size))
            elif group_name == "general":
                features[:, start:end] = self.rng.lognormal(10, 2, (n_samples, group_size))
            elif group_name in ["header", "section"]:
                features[:, start:end] = self.rng.normal(0, 1, (n_samples, group_size))
            elif group_name in ["imports", "exports"]:
                sparse_mask = self.rng.random((n_samples, group_size)) < 0.01
                features[:, start:end] = sparse_mask * self.rng.normal(0, 1, (n_samples, group_size))

        n_malicious = int(n_samples * class_balance)
        labels = np.zeros(n_samples, dtype=np.int32)
        labels[:n_malicious] = 1
        self.rng.shuffle(labels)

        return features, labels

    def load_or_generate(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        generate_if_missing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load genuine EMBER data if present, or generate synthetic data as fallback."""
        if split == "train":
            if self._check_files_exist():
                return self.load_train_data(max_samples)
            elif generate_if_missing:
                logger.warning("EMBER dataset not found, generating synthetic training data")
                return self.generate_synthetic_data(max_samples)
            else:
                raise EmberDatasetError("EMBER dataset not found and generation disabled")
        elif split == "test":
            if self._check_files_exist():
                return self.load_test_data(max_samples)
            elif generate_if_missing:
                logger.warning("EMBER dataset not found, generating synthetic test data")
                return self.generate_synthetic_data(max_samples or 200)
            else:
                raise EmberDatasetError("EMBER dataset not found and generation disabled")
        else:
            raise ValueError(f"Unknown split: {split}")

    # Legacy extractors preserved for unit-test compatibility
    def _extract_histogram(self, data: dict) -> np.ndarray:
        hist = data.get("histogram", [])
        if isinstance(hist, list) and len(hist) == 256:
            return np.array(hist, dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    def _extract_byteentropy(self, data: dict) -> np.ndarray:
        entropy = data.get("byteentropy", [])
        if isinstance(entropy, list) and len(entropy) == 256:
            return np.array(entropy, dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    def _extract_strings(self, data: dict) -> np.ndarray:
        strings_data = data.get("strings", {})
        if not strings_data:
            return np.zeros(104, dtype=np.float32)
        p_count = float(strings_data.get("printables", 0))
        div = p_count if p_count > 0 else 1.0
        p_dist = strings_data.get("printabledist", [0] * 96)
        if len(p_dist) < 96:
            p_dist = list(p_dist) + [0] * (96 - len(p_dist))
        d_norm = np.asarray(p_dist[:96], dtype=np.float32) / div
        scalars = np.array([
            strings_data.get("numstrings", 0),
            strings_data.get("avlength", 0),
            strings_data.get("printables", 0),
        ], dtype=np.float32)
        post = np.array([
            strings_data.get("entropy", 0),
            strings_data.get("paths", 0),
            strings_data.get("urls", 0),
            strings_data.get("registry", 0),
            strings_data.get("MZ", 0),
        ], dtype=np.float32)
        return np.hstack([scalars, d_norm, post])

    def _extract_general(self, data: dict) -> np.ndarray:
        general = data.get("general", {})
        features = [
            general.get("size", 0),
            general.get("vsize", 0),
            general.get("has_debug", 0),
            general.get("exports", 0),
            general.get("imports", 0),
            general.get("has_relocations", 0),
            general.get("has_resources", 0),
            general.get("has_signature", 0),
            general.get("has_tls", 0),
            general.get("symbols", 0),
        ]
        return np.array(features, dtype=np.float32)

    def _extract_header(self, data: dict) -> np.ndarray:
        header = data.get("header", {})
        optional = header.get("optional", {})
        coff = header.get("coff", {})
        features = [
            coff.get("timestamp", 0),
            coff.get("machine", 0),
            coff.get("number_of_sections", 0),
            coff.get("pointer_to_symbol_table", 0),
            coff.get("number_of_symbols", 0),
            coff.get("size_of_optional_header", 0),
            coff.get("characteristics", 0),
            *[0] * 13,
            optional.get("magic", 0),
            optional.get("major_linker_version", 0),
            optional.get("minor_linker_version", 0),
            optional.get("size_of_code", 0),
            optional.get("size_of_initialized_data", 0),
            optional.get("size_of_uninitialized_data", 0),
            optional.get("address_of_entry_point", 0),
            optional.get("base_of_code", 0),
            optional.get("base_of_data", 0),
            optional.get("image_base", 0),
            optional.get("section_alignment", 0),
            optional.get("file_alignment", 0),
            optional.get("major_os_version", 0),
            optional.get("minor_os_version", 0),
            optional.get("major_image_version", 0),
            optional.get("minor_image_version", 0),
            optional.get("major_subsystem_version", 0),
            optional.get("minor_subsystem_version", 0),
            optional.get("win32_version_value", 0),
            optional.get("size_of_image", 0),
            optional.get("size_of_headers", 0),
            optional.get("checksum", 0),
            optional.get("subsystem", 0),
            optional.get("dll_characteristics", 0),
            optional.get("size_of_stack_reserve", 0),
            optional.get("size_of_stack_commit", 0),
            optional.get("size_of_heap_reserve", 0),
            optional.get("size_of_heap_commit", 0),
            optional.get("loader_flags", 0),
            optional.get("number_of_rva_and_sizes", 0),
            *[0] * 16,
        ]
        return np.array(features[:61], dtype=np.float32)


def load_ember_data(
    data_dir: Optional[Path] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    generate_if_missing: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to load EMBER dataset."""
    loader = EmberLoader(data_dir=data_dir, random_seed=random_seed)
    return loader.load_or_generate(split, max_samples, generate_if_missing)


__all__ = [
    "EmberLoader",
    "EmberDatasetError",
    "LoaderStats",
    "iter_ember_jsonl",
    "vectorize_ember_batch",
    "vectorize_ember_record",
    "load_ember_data",
]
