"""
ThreatLens Configuration Module

Centralized configuration for EMBER 2018 dataset, feature specifications,
split ratios, random seeds, and paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Module-level constant for feature group definitions (Official EMBER 2,351 Feature Representation)
_EMBER_GROUPS: Dict[str, tuple[int, int]] = {
    "histogram": (0, 256),           # 256 features: byte histogram (0..255 counts normalized)
    "byteentropy": (256, 512),       # 256 features: 2D byte entropy histogram (16x16 joint distribution)
    "strings": (512, 616),           # 104 features: printable strings stats (numstrings, avlen, printables, 96 chars, entropy, paths, urls, registry, MZ)
    "general": (616, 626),           # 10 features: general file info (size, vsize, has_debug, exports, imports, relocs, resources, sig, tls, symbols)
    "header": (626, 688),            # 62 features: COFF & optional header (timestamp, machine 10, chars 10, subsystem 10, dll_chars 10, magic 10, + 11 scalar fields)
    "section": (688, 943),           # 255 features: section characteristics (5 general + 50 sizes + 50 entropy + 50 vsize + 50 entry + 50 chars)
    "imports": (943, 2223),          # 1280 features: imported libraries (256 bins) and imported functions (1024 bins)
    "exports": (2223, 2351),         # 128 features: exported functions (128 bins)
}


@dataclass(frozen=True)
class EmberFeatureGroups:
    """Official EMBER 2018 feature group definitions (2351 total features)."""

    TOTAL_FEATURES: int = 2351

    @classmethod
    def get_group_indices(cls, group_name: str) -> tuple[int, int]:
        """Get start and end indices for a feature group."""
        return _EMBER_GROUPS.get(group_name, (0, 0))

    @classmethod
    def get_all_group_names(cls) -> List[str]:
        """Return list of all feature group names in order."""
        return list(_EMBER_GROUPS.keys())

    @classmethod
    def validate_feature_dim(cls, n_features: int) -> bool:
        """Validate that feature dimension matches EMBER 2018 spec."""
        return n_features == cls.TOTAL_FEATURES


@dataclass(frozen=True)
class DataConfig:
    """Data paths and dataset configuration."""

    # Base directories
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    GENUINE_MODELS_DIR: Path = ARTIFACTS_DIR / "models" / "ember2018_genuine"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # EMBER 2018 specific
    EMBER_VERSION: str = "2018"
    EMBER_URL: str = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
    EMBER_ARCHIVE_NAME: str = "ember_dataset_2018_2.tar.bz2"

    # Expected EMBER 2018 files after extraction
    EMBER_TRAIN_FEATURES: str = "train_features.jsonl"
    EMBER_TRAIN_LABELS: str = "train_labels.csv"
    EMBER_TEST_FEATURES: str = "test_features.jsonl"
    EMBER_TEST_LABELS: str = "test_labels.csv"
    EMBER_TRAIN_FEATURE_FILES: tuple[str, ...] = (
        "train_features_0.jsonl",
        "train_features_1.jsonl",
        "train_features_2.jsonl",
        "train_features_3.jsonl",
        "train_features_4.jsonl",
        "train_features_5.jsonl",
    )

    # Synthetic data (for testing when EMBER not available)
    SYNTHETIC_SAMPLES: int = 1000
    SYNTHETIC_FEATURE_DIM: int = EmberFeatureGroups.TOTAL_FEATURES

    def get_ember_raw_dir(self) -> Path:
        """Return the directory containing extracted EMBER files."""
        ember_subdir = self.RAW_DATA_DIR / "ember2018"
        if ember_subdir.exists():
            return ember_subdir
        return self.RAW_DATA_DIR

    def __post_init__(self):
        """Create directories on initialization."""
        for dir_path in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.ARTIFACTS_DIR,
            self.MODELS_DIR,
            self.GENUINE_MODELS_DIR,
            self.LOGS_DIR,
            self.REPORTS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SplitConfig:
    """Train/Validation/Calibration/Test split configuration."""

    # Split ratios (must sum to 1.0)
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.15
    CAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.10

    # Random seed for reproducibility
    RANDOM_SEED: int = 42

    # Stratify by label
    STRATIFY: bool = True

    def __post_init__(self):
        """Validate split ratios sum to 1.0."""
        total = self.TRAIN_RATIO + self.VAL_RATIO + self.CAL_RATIO + self.TEST_RATIO
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""

    # Imputation strategy for missing values
    IMPUTE_STRATEGY: str = "median"  # 'mean', 'median', 'constant'
    IMPUTE_CONSTANT: float = 0.0

    # Scaling
    SCALER_TYPE: str = "standard"  # 'standard', 'robust', 'minmax', None
    WITH_CENTERING: bool = True
    WITH_SCALING: bool = True

    # Feature selection (None = keep all EMBER features)
    FEATURE_SELECTION: Optional[str] = None  # 'variance', 'mutual_info', None

    # Output
    SAVE_PREPROCESSOR: bool = True
    PREPROCESSOR_FILENAME: str = "preprocessor.joblib"

    # Random seed
    RANDOM_SEED: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """LightGBM binary classifier configuration for EMBER 2018 model."""

    BOOSTING_TYPE: str = "gbdt"
    OBJECTIVE: str = "binary"
    METRIC: tuple[str, ...] = ("binary_logloss", "auc")
    LEARNING_RATE: float = 0.05
    NUM_LEAVES: int = 31
    MAX_DEPTH: int = -1
    MIN_DATA_IN_LEAF: int = 20
    FEATURE_FRACTION: float = 0.8
    BAGGING_FRACTION: float = 0.8
    BAGGING_FREQ: int = 1
    NUM_BOOST_ROUND: int = 1000
    EARLY_STOPPING_ROUNDS: int = 50
    RANDOM_SEED: int = 42
    IS_UNBALANCE: bool = False
    VERBOSITY: int = -1
    MODEL_FILENAME: str = "lightgbm_ember.txt"
    METADATA_FILENAME: str = "model_metadata.json"


@dataclass(frozen=True)
class CalibrationConfig:
    """Probability calibration configuration."""

    METHOD: str = "sigmoid"  # 'sigmoid' (Platt scaling) or 'isotonic'
    CALIBRATOR_FILENAME: str = "calibrator.joblib"
    METADATA_FILENAME: str = "calibration_metadata.json"
    RANDOM_SEED: int = 42
    DEFAULT_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class APIConfig:
    """FastAPI configuration."""

    TITLE: str = "ThreatLens Static Malware Detection API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Explainable AI-Powered Static Malware Detection and Analysis System for Windows PE files"
    )
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    MAX_BATCH_SIZE: int = 100
    DEFAULT_TOP_K: int = 20
    MAX_TOP_K: int = 100
    CORS_ORIGINS: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class DashboardConfig:
    """Streamlit dashboard configuration."""

    DEFAULT_API_URL: str = "http://127.0.0.1:8000"
    API_TIMEOUT: float = 10.0
    CONNECT_TIMEOUT: float = 3.0
    MAX_UPLOAD_SIZE_MB: int = 50
    DEFAULT_TOP_K_EXPLAIN: int = 15
    THEME_DARK: bool = True
    MAX_TEST_SAMPLES_DISPLAY: int = 10


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    LEVEL: str = "INFO"
    FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: str = "threatlens.log"
    MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5


# Global configuration instances
DATA_CONFIG = DataConfig()
SPLIT_CONFIG = SplitConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
MODEL_CONFIG = ModelConfig()
CALIBRATION_CONFIG = CalibrationConfig()
API_CONFIG = APIConfig()
DASHBOARD_CONFIG = DashboardConfig()
LOGGING_CONFIG = LoggingConfig()
FEATURE_GROUPS = EmberFeatureGroups()

# Convenience exports
__all__ = [
    "DATA_CONFIG",
    "SPLIT_CONFIG",
    "PREPROCESSING_CONFIG",
    "MODEL_CONFIG",
    "CALIBRATION_CONFIG",
    "API_CONFIG",
    "DASHBOARD_CONFIG",
    "LOGGING_CONFIG",
    "FEATURE_GROUPS",
    "EmberFeatureGroups",
    "DataConfig",
    "SplitConfig",
    "PreprocessingConfig",
    "ModelConfig",
    "CalibrationConfig",
    "APIConfig",
    "DashboardConfig",
    "LoggingConfig",
]