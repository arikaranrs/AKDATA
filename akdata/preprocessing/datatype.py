"""DataType Caster Preprocessor Module for AKDATA.

Auto-detects semantic types and casts columns to numerical, categorical, or datetime.
"""

import pandas as pd
from typing import Any, Dict
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class DataTypeCaster(BaseTransformer):
    """Automatically casts columns to their most appropriate data types.

    Infers numeric, datetime, and categorical columns.
    Saves datatype mapping during fit to apply during transform.
    """

    def __init__(self, category_cardinality_threshold: float = 0.05):
        """Initialize the DataTypeCaster.

        Args:
            category_cardinality_threshold (float): Cardinality ratio (unique / total)
                below which an object column is cast to category.
        """
        super().__init__()
        self.category_cardinality_threshold = category_cardinality_threshold
        self.type_mapping: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "DataTypeCaster":
        """Determine target datatype casting for each column.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            DataTypeCaster: Self instance.
        """
        logger.info("Fitting DataTypeCaster...")
        self.type_mapping = {}
        num_rows = len(df)

        for col in df.columns:
            series = df[col]
            current_dtype = series.dtype

            # 1. Already numeric or datetime
            if pd.api.types.is_numeric_dtype(series):
                # Check if it's float or int, keep it
                self.type_mapping[col] = str(current_dtype)
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                self.type_mapping[col] = "datetime64[ns]"
                continue

            # 2. Object or category columns
            non_null = series.dropna()
            if len(non_null) == 0:
                self.type_mapping[col] = "object"
                continue

            # Check if it could be datetime
            # Attempt to convert a sample to datetime. If it succeeds, mark as datetime.
            try:
                # Try sample of up to 100 values
                sample = non_null.head(100)
                converted_sample = pd.to_datetime(sample, errors="raise")
                # Ensure it's not parsing plain numbers as timestamps (like 2023)
                if not all(isinstance(val, (int, float)) or str(val).isdigit() for val in sample):
                    # Safe to assume it is a date
                    self.type_mapping[col] = "datetime64[ns]"
                    continue
            except (ValueError, TypeError):
                pass

            # Check if it could be numeric
            try:
                pd.to_numeric(non_null, errors="raise")
                self.type_mapping[col] = "float64"  # Default to float for safety with nulls
                continue
            except (ValueError, TypeError):
                pass

            # Check for Categorical based on cardinality ratio
            unique_count = non_null.nunique()
            cardinality_ratio = unique_count / num_rows if num_rows > 0 else 1.0
            if cardinality_ratio < self.category_cardinality_threshold or unique_count < 10:
                self.type_mapping[col] = "category"
            else:
                self.type_mapping[col] = "string"

        self.meta_info["type_mapping"] = self.type_mapping
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to the types identified during fitting.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with cast column types.
        """
        logger.info("Executing DataTypeCaster transform...")
        current_df = df.copy()

        for col, target_type in self.type_mapping.items():
            if col in current_df.columns:
                try:
                    if target_type == "datetime64[ns]":
                        current_df[col] = pd.to_datetime(current_df[col], errors="coerce")
                    elif target_type in ("float64", "int64"):
                        current_df[col] = pd.to_numeric(current_df[col], errors="coerce")
                    elif target_type == "category":
                        current_df[col] = current_df[col].astype("category")
                    elif target_type == "string":
                        current_df[col] = current_df[col].astype(str)
                    logger.debug(f"Casted column '{col}' to type '{target_type}'.")
                except Exception as e:
                    logger.warning(f"Failed to cast column '{col}' to '{target_type}': {e}")

        return current_df
