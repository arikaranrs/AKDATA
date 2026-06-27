"""Cleaning Preprocessor Module for AKDATA.

Standardizes column names, trims spaces, and drops empty rows/columns.
"""

import re
import pandas as pd
from typing import Any, Dict
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class DataCleaner(BaseTransformer):
    """Cleaner for standardizing column names and cell string values.

    Drops columns and rows that are completely empty.
    """

    def __init__(self):
        super().__init__()
        self.column_mapping: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "DataCleaner":
        """Identify which columns need renaming and which columns are completely empty.

        Args:
            df (pd.DataFrame): Input DataFrame.
            y (Any, optional): Target vector. Defaults to None.

        Returns:
            DataCleaner: Self instance.
        """
        logger.info("Fitting DataCleaner...")
        self.column_mapping = {}

        for col in df.columns:
            # Lowercase, replace spaces/hyphens with underscore, strip special chars
            clean_name = col.strip().lower()
            clean_name = re.sub(r"[\s\-]+", "_", clean_name)
            clean_name = re.sub(r"[^\w]+", "", clean_name)
            self.column_mapping[col] = clean_name

        # Detect completely empty columns
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        self.meta_info["renamed_columns"] = {k: v for k, v in self.column_mapping.items() if k != v}
        self.meta_info["empty_columns_dropped"] = empty_cols
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names, trim strings, and drop empty rows/columns.

        Args:
            df (pd.DataFrame): DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Executing DataCleaner transform...")
        current_df = df.copy()

        # Renaming columns
        current_df = current_df.rename(columns=self.column_mapping)

        # Drop empty columns
        empty_cols_in_df = [self.column_mapping[c] for c in self.meta_info["empty_columns_dropped"] if c in self.column_mapping]
        current_df = current_df.drop(columns=empty_cols_in_df)

        # Drop completely empty rows
        initial_rows = len(current_df)
        current_df = current_df.dropna(how="all")
        dropped_rows = initial_rows - len(current_df)
        self.meta_info["empty_rows_dropped"] = dropped_rows

        # Trim spaces in string cells
        string_cols = current_df.select_dtypes(include=["object", "string"])
        for col in string_cols.columns:
            try:
                current_df[col] = current_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            except Exception as e:
                logger.warning(f"Failed to strip strings in column {col}: {e}")

        logger.info(f"Data cleaning complete. Dropped {len(empty_cols_in_df)} columns and {dropped_rows} rows.")
        return current_df
