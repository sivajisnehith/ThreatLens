"""
EMBER 2018 Dataset Loader

Handles downloading, extracting, and loading the EMBER 2018 dataset.
Supports both the official precomputed features (JSONL format) and
generates synthetic data for testing when the dataset is unavailable.
"""

import json
import logging
import os
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from threatlens.config import DATA_CONFIG, FEATURE_GROUPS, EmberFeatureGroups
from threatlens.utils.logger import get_logger

logger = get_logger("ember_loader")


class EmberDatasetError(Exception):
    """Custom exception for EMBER dataset operations."""
    pass


class EmberLoader:
    """
    Loader for EMBER 2018 dataset.

    EMBER 2018 provides precomputed features for 1,100,000 samples
    (900,000 train + 200,000 test) with 2351 features each.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the EMBER loader.

        Args:
            data_dir: Directory containing EMBER dataset files
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir or DATA_CONFIG.RAW_DATA_DIR
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Expected file paths
        self.train_features_path = self.data_dir / DATA_CONFIG.EMBER_TRAIN_FEATURES
        self.train_labels_path = self.data_dir / DATA_CONFIG.EMBER_TRAIN_LABELS
        self.test_features_path = self.data_dir / DATA_CONFIG.EMBER_TEST_FEATURES
        self.test_labels_path = self.data_dir / DATA_CONFIG.EMBER_TEST_LABELS

    def download_dataset(self, force: bool = False) -> bool:
        """
        Download EMBER 2018 dataset from official source.

        Args:
            force: Force re-download even if files exist

        Returns:
            True if download successful, False otherwise
        """
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
        """Extract the EMBER tar.bz2 archive."""
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=self.data_dir)

        # Verify extraction
        if not self._check_files_exist():
            raise EmberDatasetError("Extraction completed but expected files not found")

    def _check_files_exist(self) -> bool:
        """Check if all expected EMBER files exist."""
        required_files = [
            self.train_features_path,
            self.train_labels_path,
            self.test_features_path,
            self.test_labels_path,
        ]
        return all(f.exists() for f in required_files)

    def load_raw_jsonl(self, filepath: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load EMBER JSONL feature file.

        Args:
            filepath: Path to JSONL file
            max_samples: Maximum number of samples to load (for testing)

        Returns:
            DataFrame with features
        """
        logger.info(f"Loading features from {filepath}")

        features_list = []
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    features_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {i}: {e}")

        df = pd.DataFrame(features_list)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} feature columns")
        return df

    def load_labels(self, filepath: Path) -> pd.Series:
        """
        Load EMBER labels CSV file.

        Args:
            filepath: Path to labels CSV

        Returns:
            Series with labels (0=benign, 1=malicious, -1=unlabeled)
        """
        logger.info(f"Loading labels from {filepath}")
        labels_df = pd.read_csv(filepath)

        # EMBER labels CSV has columns: sha256, label
        if "label" not in labels_df.columns:
            raise EmberDatasetError(f"Labels file missing 'label' column: {filepath}")

        labels = labels_df["label"].astype(int)
        logger.info(f"Loaded {len(labels)} labels. Distribution: {labels.value_counts().to_dict()}")
        return labels

    def load_train_data(
        self,
        max_samples: Optional[int] = None,
        include_unlabeled: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training features and labels as numpy arrays.

        Args:
            max_samples: Maximum samples to load (for testing)
            include_unlabeled: Whether to include unlabeled samples (label=-1)

        Returns:
            Tuple of (features_array, labels_array)
        """
        features_df = self.load_raw_jsonl(self.train_features_path, max_samples)
        labels_series = self.load_labels(self.train_labels_path)

        # Align by index (EMBER files are ordered consistently)
        if len(features_df) != len(labels_series):
            logger.warning(
                f"Feature count ({len(features_df)}) != Label count ({len(labels_series)})"
            )
            min_len = min(len(features_df), len(labels_series))
            features_df = features_df.iloc[:min_len]
            labels_series = labels_series.iloc[:min_len]

        # Filter unlabeled if requested
        if not include_unlabeled:
            labeled_mask = labels_series != -1
            features_df = features_df[labeled_mask]
            labels_series = labels_series[labeled_mask]
            logger.info(f"Filtered to {len(labels_series)} labeled samples")

        # Convert features to numpy array (2351-dimensional vectors)
        features = self._features_df_to_array(features_df)
        labels = labels_series.values.astype(np.int32)

        return features, labels

    def load_test_data(
        self,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test features and labels as numpy arrays.

        Args:
            max_samples: Maximum samples to load (for testing)

        Returns:
            Tuple of (features_array, labels_array)
        """
        features_df = self.load_raw_jsonl(self.test_features_path, max_samples)
        labels_series = self.load_labels(self.test_labels_path)

        if len(features_df) != len(labels_series):
            logger.warning(
                f"Feature count ({len(features_df)}) != Label count ({len(labels_series)})"
            )
            min_len = min(len(features_df), len(labels_series))
            features_df = features_df.iloc[:min_len]
            labels_series = labels_series.iloc[:min_len]

        features = self._features_df_to_array(features_df)
        labels = labels_series.values.astype(np.int32)

        return features, labels

    def _features_df_to_array(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Convert EMBER feature DataFrame to fixed-size numpy array.

        EMBER features are stored as variable-length dicts in JSONL.
        This method extracts the 'histogram', 'byteentropy', 'strings',
        'general', 'header', 'section', 'imports', 'exports' feature groups
        and concatenates them into a 2351-dimensional vector per sample.
        """
        n_samples = len(features_df)
        n_features = EmberFeatureGroups.TOTAL_FEATURES
        feature_array = np.zeros((n_samples, n_features), dtype=np.float32)

        # Feature group extraction functions
        group_extractors = {
            "histogram": lambda x: self._extract_histogram(x),
            "byteentropy": lambda x: self._extract_byteentropy(x),
            "strings": lambda x: self._extract_strings(x),
            "general": lambda x: self._extract_general(x),
            "header": lambda x: self._extract_header(x),
            "section": lambda x: self._extract_section(x),
            "imports": lambda x: self._extract_imports(x),
            "exports": lambda x: self._extract_exports(x),
        }

        for idx, row in features_df.iterrows():
            row_dict = row.to_dict() if isinstance(row, pd.Series) else row
            feature_vector = np.zeros(n_features, dtype=np.float32)

            for group_name, extractor in group_extractors.items():
                start, end = FEATURE_GROUPS.get_group_indices(group_name)
                group_data = row_dict.get(group_name, {})
                try:
                    extracted = extractor(group_data)
                    if len(extracted) == (end - start):
                        feature_vector[start:end] = extracted
                    else:
                        logger.debug(
                            f"Sample {idx}: {group_name} length mismatch "
                            f"({len(extracted)} vs {end-start})"
                        )
                except Exception as e:
                    logger.debug(f"Sample {idx}: Failed to extract {group_name}: {e}")

            feature_array[idx] = feature_vector

        return feature_array

    # Feature group extractors matching EMBER 2018 specification
    def _extract_histogram(self, data: dict) -> np.ndarray:
        """Extract byte histogram (256 features)."""
        hist = data.get("histogram", [])
        if isinstance(hist, list) and len(hist) == 256:
            return np.array(hist, dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    def _extract_byteentropy(self, data: dict) -> np.ndarray:
        """Extract byte entropy histogram (256 features)."""
        entropy = data.get("byteentropy", [])
        if isinstance(entropy, list) and len(entropy) == 256:
            return np.array(entropy, dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    def _extract_strings(self, data: dict) -> np.ndarray:
        """Extract string statistics (256 features)."""
        # EMBER strings: [printables, entropy, paths, urls, registry, mz]
        # Each is a histogram-like feature
        strings_data = data.get("strings", {})
        if not strings_data:
            return np.zeros(256, dtype=np.float32)

        # Concatenate all string histograms
        parts = []
        for key in ["printables", "entropy", "paths", "urls", "registry", "mz"]:
            val = strings_data.get(key, [])
            if isinstance(val, list):
                parts.extend(val)
        arr = np.array(parts[:256], dtype=np.float32)
        if len(arr) < 256:
            arr = np.pad(arr, (0, 256 - len(arr)))
        return arr[:256]

    def _extract_general(self, data: dict) -> np.ndarray:
        """Extract general file info (10 features)."""
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
        """Extract COFF header features (61 features)."""
        header = data.get("header", {})
        optional = header.get("optional", {})
        coff = header.get("coff", {})

        features = [
            # COFF header (20 features)
            coff.get("timestamp", 0),
            coff.get("machine", 0),
            coff.get("number_of_sections", 0),
            coff.get("pointer_to_symbol_table", 0),
            coff.get("number_of_symbols", 0),
            coff.get("size_of_optional_header", 0),
            coff.get("characteristics", 0),
            # Pad to 20
            *[0] * 13,
            # Optional header (41 features)
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
            # Data directories (16 features - 8 dirs * 2)
            *[0] * 16,
        ]
        arr = np.array(features[:61], dtype=np.float32)
        if len(arr) < 61:
            arr = np.pad(arr, (0, 61 - len(arr)))
        return arr[:61]

    def _extract_section(self, data: dict) -> np.ndarray:
        """Extract section features (196 features)."""
        sections = data.get("section", {})
        # EMBER sections: list of dicts with entropy, size, etc.
        section_list = sections.get("sections", []) if isinstance(sections, dict) else sections

        # Each section: entropy, size, virtual_size, characteristics, name_hash
        # Max ~10 sections * ~20 features = ~200 features
        features = []
        if isinstance(section_list, list):
            for sec in section_list[:10]:  # Max 10 sections
                if isinstance(sec, dict):
                    features.extend([
                        sec.get("entropy", 0),
                        sec.get("size", 0),
                        sec.get("virtual_size", 0),
                        sec.get("characteristics", 0),
                        sec.get("name_hash", 0),
                    ])
        arr = np.array(features[:196], dtype=np.float32)
        if len(arr) < 196:
            arr = np.pad(arr, (0, 196 - len(arr)))
        return arr[:196]

    def _extract_imports(self, data: dict) -> np.ndarray:
        """Extract imports features (1248 features)."""
        imports = data.get("imports", {})
        # EMBER imports: dict of dll -> list of imported functions
        # Features are hashed function names (feature hashing trick)
        imports_list = imports.get("imports", []) if isinstance(imports, dict) else imports

        features = []
        if isinstance(imports_list, list):
            for imp in imports_list:
                if isinstance(imp, dict):
                    # Hashed import names
                    features.extend([
                        imp.get("hash", 0),
                        imp.get("dll_hash", 0),
                    ])
        arr = np.array(features[:1248], dtype=np.float32)
        if len(arr) < 1248:
            arr = np.pad(arr, (0, 1248 - len(arr)))
        return arr[:1248]

    def _extract_exports(self, data: dict) -> np.ndarray:
        """Extract exports features (68 features)."""
        exports = data.get("exports", {})
        exports_list = exports.get("exports", []) if isinstance(exports, dict) else exports

        features = []
        if isinstance(exports_list, list):
            for exp in exports_list:
                if isinstance(exp, dict):
                    features.extend([
                        exp.get("hash", 0),
                        exp.get("ordinal", 0),
                    ])
        arr = np.array(features[:68], dtype=np.float32)
        if len(arr) < 68:
            arr = np.pad(arr, (0, 68 - len(arr)))
        return arr[:68]

    def generate_synthetic_data(
        self,
        n_samples: Optional[int] = None,
        class_balance: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EMBER-like data for testing.

        Args:
            n_samples: Number of samples to generate
            class_balance: Proportion of malicious samples (0-1)

        Returns:
            Tuple of (features_array, labels_array)
        """
        n_samples = n_samples or DATA_CONFIG.SYNTHETIC_SAMPLES
        n_features = FEATURE_GROUPS.TOTAL_FEATURES

        logger.info(f"Generating {n_samples} synthetic samples with {n_features} features")

        # Generate features with realistic distributions per group
        features = np.zeros((n_samples, n_features), dtype=np.float32)

        for group_name in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(group_name)
            group_size = end - start

            if group_name in ["histogram", "byteentropy"]:
                # Histogram-like: Dirichlet distribution (sums to ~1)
                alpha = self.rng.uniform(0.1, 1.0, group_size)
                features[:, start:end] = self.rng.dirichlet(alpha, n_samples)
            elif group_name == "strings":
                # String features: sparse, mostly zeros
                sparse_mask = self.rng.random((n_samples, group_size)) < 0.05
                features[:, start:end] = sparse_mask * self.rng.exponential(1.0, (n_samples, group_size))
            elif group_name == "general":
                # General: file size, counts
                features[:, start:end] = self.rng.lognormal(10, 2, (n_samples, group_size))
            elif group_name in ["header", "section"]:
                # Header/Section: mixed categorical and continuous
                features[:, start:end] = self.rng.normal(0, 1, (n_samples, group_size))
            elif group_name in ["imports", "exports"]:
                # Imports/Exports: sparse hashed features
                sparse_mask = self.rng.random((n_samples, group_size)) < 0.01
                features[:, start:end] = sparse_mask * self.rng.normal(0, 1, (n_samples, group_size))

        # Generate labels with specified balance
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
        """
        Load EMBER data or generate synthetic if not available.

        Args:
            split: 'train' or 'test'
            max_samples: Maximum samples to load
            generate_if_missing: Generate synthetic data if EMBER not found

        Returns:
            Tuple of (features, labels)
        """
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


def load_ember_data(
    data_dir: Optional[Path] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    generate_if_missing: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load EMBER dataset.

    Args:
        data_dir: Directory containing EMBER files
        split: 'train' or 'test'
        max_samples: Maximum samples to load
        generate_if_missing: Generate synthetic if EMBER not found
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels)
    """
    loader = EmberLoader(data_dir=data_dir, random_seed=random_seed)
    return loader.load_or_generate(split, max_samples, generate_if_missing)


__all__ = [
    "EmberLoader",
    "EmberDatasetError",
    "load_ember_data",
]