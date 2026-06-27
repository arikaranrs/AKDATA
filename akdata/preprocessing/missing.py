"""Missing Value Imputation Preprocessor Module for AKDATA.

Imputes missing values for numerical and categorical columns.
"""

import pandas as pd
from typing import Any, Dict, Optional
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class MissingValueImputer(BaseTransformer):
    """Imputes missing values in numerical and categorical columns.

    Saves the imputed values from training data to apply consistently on test data.
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        fill_value: Any = None
    ):
        """Initialize the imputer.

        Args:
            numeric_strategy (str): Imputation strategy for numeric columns ('mean', 'median', 'mode', 'constant').
            categorical_strategy (str): Imputation strategy for categorical columns ('mode', 'constant').
            fill_value (Any, optional): Constant value to use if strategy is 'constant'.
        """
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.impute_values: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "MissingValueImputer":
        """Compute the fill values for missing columns.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            MissingValueImputer: Self instance.
        """
        logger.info("Fitting MissingValueImputer...")
        self.impute_values = {}

        for col in df.columns:
            series = df[col]
            # Skip if there are no nulls, but compute just in case test data has nulls
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) == 0:
                    val = 0.0 if self.fill_value is None else self.fill_value
                elif self.numeric_strategy == "mean":
                    val = float(non_null.mean())
                elif self.numeric_strategy == "median":
                    val = float(non_null.median())
                elif self.numeric_strategy == "mode":
                    val = float(non_null.mode()[0])
                elif self.numeric_strategy == "constant":
                    val = 0.0 if self.fill_value is None else self.fill_value
                else:
                    logger.warning(f"Unknown numeric strategy '{self.numeric_strategy}'. Defaulting to median.")
                    val = float(non_null.median())
                self.impute_values[col] = val

            else:
                non_null = series.dropna()
                if len(non_null) == 0:
                    val = "missing" if self.fill_value is None else str(self.fill_value)
                elif self.categorical_strategy == "mode":
                    val = str(non_null.mode()[0])
                elif self.categorical_strategy == "constant":
                    val = "missing" if self.fill_value is None else str(self.fill_value)
                else:
                    logger.warning(f"Unknown categorical strategy '{self.categorical_strategy}'. Defaulting to mode.")
                    val = str(non_null.mode()[0])
                self.impute_values[col] = val

        self.meta_info["impute_values"] = self.impute_values
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to impute.

        Returns:
            pd.DataFrame: DataFrame with filled missing values.
        """
        logger.info("Executing MissingValueImputer transform...")
        current_df = df.copy()

        for col in current_df.columns:
            if col in self.impute_values:
                # Count missing entries beforehand
                null_count = int(current_df[col].isnull().sum())
                if null_count > 0:
                    fill_val = self.impute_values[col]
                    current_df[col] = current_df[col].fillna(fill_val)
                    logger.debug(f"Imputed {null_count} missing values in column '{col}' with: {fill_val}")

        return current_df
