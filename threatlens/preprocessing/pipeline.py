"""
EMBER Feature Preprocessing Pipeline

Implements imputation and scaling fitted ONLY on training data to prevent
data leakage, while preserving the exact 2351 EMBER 2018 feature structure.
"""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from threatlens.config import DATA_CONFIG, PREPROCESSING_CONFIG, PreprocessingConfig
from threatlens.utils.logger import get_logger

logger = get_logger("preprocessing")


class EmberPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for EMBER 2018 features.

    Applies imputation and optional scaling, maintaining strict feature alignment
    and random reproducibility.
    """

    def __init__(self, config: PreprocessingConfig = PREPROCESSING_CONFIG):
        """
        Initialize preprocessor.

        Args:
            config: PreprocessingConfig instance
        """
        self.config = config
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None
        self.is_fitted: bool = False

        self._build_components()

    def _build_components(self) -> None:
        """Instantiate imputer and scaler based on config."""
        # Imputer
        strategy = self.config.IMPUTE_STRATEGY
        fill_val = self.config.IMPUTE_CONSTANT if strategy == "constant" else None
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_val)

        # Scaler
        scaler_type = self.config.SCALER_TYPE
        if scaler_type == "standard":
            self.scaler = StandardScaler(
                with_mean=self.config.WITH_CENTERING,
                with_std=self.config.WITH_SCALING,
            )
        elif scaler_type == "robust":
            self.scaler = RobustScaler(
                with_centering=self.config.WITH_CENTERING,
                with_scaling=self.config.WITH_SCALING,
            )
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type is None or scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EmberPreprocessor":
        """
        Fit imputer and scaler ONLY on training features X.

        Args:
            X: Training features array of shape (N, D)
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        logger.info(f"Fitting preprocessor on training data with shape {X.shape}...")

        # Convert Inf values to NaN prior to fitting imputer
        X_clean = np.where(np.isinf(X), np.nan, X)

        # Fit imputer
        X_imputed = self.imputer.fit_transform(X_clean)

        # Fit scaler if configured
        if self.scaler is not None:
            self.scaler.fit(X_imputed)

        self.is_fitted = True
        logger.info("Preprocessor fit completed successfully")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features X using fitted imputer and scaler.

        Args:
            X: Features array of shape (N, D)

        Returns:
            Transformed features array
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        X_clean = np.where(np.isinf(X), np.nan, X)
        X_imputed = self.imputer.transform(X_clean)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
            return X_scaled.astype(np.float32)

        return X_imputed.astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit preprocessor and transform training features X."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save fitted preprocessor to disk.

        Args:
            filepath: Path to save file (default: ARTIFACTS_DIR / preprocessor.joblib)

        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        save_path = filepath or (DATA_CONFIG.ARTIFACTS_DIR / self.config.PREPROCESSOR_FILENAME)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, save_path)
        logger.info(f"Saved fitted preprocessor to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Path) -> "EmberPreprocessor":
        """
        Load fitted preprocessor from disk.

        Args:
            filepath: Path to saved preprocessor joblib file

        Returns:
            Fitted EmberPreprocessor instance
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {filepath}")

        preprocessor = joblib.load(filepath)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}")

        logger.info(f"Loaded preprocessor from {filepath}")
        return preprocessor


__all__ = ["EmberPreprocessor"]