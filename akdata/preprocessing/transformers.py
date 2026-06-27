"""Mathematical Transformers Module for AKDATA.

Applies log, square root, or exponential transformations to numeric columns.
Handles negative values and zeros safely.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class MathTransformer(BaseTransformer):
    """Applies mathematical transformations to numerical columns.

    Supported functions: 'log', 'log1p', 'sqrt', 'exp'.
    """

    def __init__(self, transformations: Optional[Dict[str, str]] = None):
        """Initialize MathTransformer.

        Args:
            transformations (Optional[Dict[str, str]]): Dictionary mapping column names to
                transformation strategy ('log', 'log1p', 'sqrt', 'exp').
        """
        super().__init__()
        self.transformations = transformations if transformations is not None else {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "MathTransformer":
        """Validate that all target columns for transformation exist in training data.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            MathTransformer: Self instance.
        """
        logger.info("Fitting MathTransformer...")
        # Validate that specified columns are numerical
        self.meta_info["transformations_applied"] = {}
        for col, method in self.transformations.items():
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.meta_info["transformations_applied"][col] = method
                else:
                    logger.warning(f"Math transformation requested for non-numeric column '{col}'. Skipping.")
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mathematical transformations.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Executing MathTransformer transform...")
        current_df = df.copy()

        for col, method in self.meta_info.get("transformations_applied", {}).items():
            if col in current_df.columns:
                method_lower = method.lower()
                series = current_df[col]

                try:
                    if method_lower == "log":
                        # Safe log: offset to make min value > 0 if there are negative or zero values
                        min_val = series.min()
                        offset = 0.0
                        if min_val <= 0:
                            offset = abs(min_val) + 1.0
                            logger.info(f"Adding offset {offset:.2f} to column '{col}' to allow safe log.")
                        current_df[col] = np.log(series + offset)

                    elif method_lower == "log1p":
                        min_val = series.min()
                        offset = 0.0
                        if min_val < -1.0:
                            offset = abs(min_val) - 0.99
                            logger.info(f"Adding offset {offset:.2f} to column '{col}' to allow safe log1p.")
                        current_df[col] = np.log1p(series + offset)

                    elif method_lower == "sqrt":
                        # Safe sqrt: set negative values to zero or take absolute
                        min_val = series.min()
                        if min_val < 0:
                            logger.info(f"Replacing negative values in column '{col}' with 0 for square root.")
                            series_clipped = series.clip(lower=0)
                            current_df[col] = np.sqrt(series_clipped)
                        else:
                            current_df[col] = np.sqrt(series)

                    elif method_lower == "exp":
                        # Prevent overflow by clipping large values
                        # e.g. e^88 starts exceeding max float bounds
                        series_clipped = series.clip(upper=50)
                        current_df[col] = np.exp(series_clipped)

                    logger.debug(f"Applied mathematical transformation '{method_lower}' to column '{col}'.")
                except Exception as e:
                    logger.warning(f"Failed to apply '{method}' transformation to column '{col}': {e}")

        return current_df
