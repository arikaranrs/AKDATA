"""Scaling Preprocessor Module for AKDATA.

Performs MinMax or Standard scaling on numerical columns.
Saves fit parameters (min, max, mean, std) to scale test sets without data leakage.
"""

import pandas as pd
from typing import Any, Dict
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class NumericScaler(BaseTransformer):
    """Scales numerical columns using Standard or MinMax scaling."""

    def __init__(self, strategy: str = "standard"):
        """Initialize the NumericScaler.

        Args:
            strategy (str): Scaling strategy: 'standard' or 'minmax'.
        """
        super().__init__()
        self.strategy = strategy.lower()
        self.params: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "NumericScaler":
        """Learn scaling parameters (mean/std or min/max) from numerical columns.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            NumericScaler: Self instance.
        """
        logger.info(f"Fitting NumericScaler (strategy={self.strategy})...")
        self.params = {}
        numeric_cols = df.select_dtypes(include=["number"])

        for col in numeric_cols.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if self.strategy == "standard":
                mean_val = float(series.mean())
                std_val = float(series.std())
                # Handle zero variance
                if std_val == 0.0:
                    std_val = 1.0
                self.params[col] = {"mean": mean_val, "std": std_val}

            elif self.strategy == "minmax":
                min_val = float(series.min())
                max_val = float(series.max())
                diff = max_val - min_val
                # Handle zero variance
                if diff == 0.0:
                    diff = 1.0
                self.params[col] = {"min": min_val, "range": diff}
            else:
                logger.warning(f"Unknown scaling strategy '{self.strategy}'. Defaulting to standard.")
                mean_val = float(series.mean())
                std_val = float(series.std())
                if std_val == 0.0:
                    std_val = 1.0
                self.params[col] = {"mean": mean_val, "std": std_val}

        self.meta_info["scaling_params"] = self.params
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to the DataFrame using parameters computed in fit.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        logger.info("Executing NumericScaler transform...")
        current_df = df.copy()

        for col, col_params in self.params.items():
            if col in current_df.columns:
                try:
                    if self.strategy == "standard":
                        mean = col_params["mean"]
                        std = col_params["std"]
                        current_df[col] = (current_df[col] - mean) / std
                    elif self.strategy == "minmax":
                        min_val = col_params["min"]
                        range_val = col_params["range"]
                        current_df[col] = (current_df[col] - min_val) / range_val
                    logger.debug(f"Scaled column '{col}' using {self.strategy}.")
                except Exception as e:
                    logger.warning(f"Failed to scale column '{col}': {e}")

        return current_df
