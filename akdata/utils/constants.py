"""Constants used across the AKDATA library.

Provides default thresholds, values, and configurations.
"""

from typing import Final

# Random state default
DEFAULT_RANDOM_STATE: Final[int] = 42

# Preprocessing thresholds
CORRELATION_THRESHOLD: Final[float] = 0.90  # For target leakage or redundant feature selection
VARIANCE_THRESHOLD: Final[float] = 0.01     # For feature selection
OUTLIER_IQR_FACTOR: Final[float] = 1.5      # For IQR outlier detection
OUTLIER_Z_SCORE_THRESHOLD: Final[float] = 3.0  # For Z-score outlier detection

# Health Score weights
WEIGHT_MISSING: Final[float] = 0.4
WEIGHT_DUPLICATES: Final[float] = 0.3
WEIGHT_OUTLIERS: Final[float] = 0.2
WEIGHT_TYPE_CONFLICTS: Final[float] = 0.1

# Default missing value fill values
DEFAULT_NUMERIC_IMPUTATION: Final[str] = "median"
DEFAULT_CATEGORICAL_IMPUTATION: Final[str] = "mode"
