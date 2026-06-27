"""Duplicate Handler Preprocessor Module for AKDATA.

Identifies and removes duplicate rows and duplicate columns.
"""

import pandas as pd
from typing import Any, List
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class DuplicateHandler(BaseTransformer):
    """Detects and drops duplicate rows and columns from the dataset.

    Maintains integrity by tracking duplicate columns during training.
    """

    def __init__(self):
        super().__init__()
        self.duplicate_columns: List[str] = []

    def fit(self, df: pd.DataFrame, y: Any = None) -> "DuplicateHandler":
        """Identify duplicate columns in the DataFrame.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            DuplicateHandler: Self instance.
        """
        logger.info("Fitting DuplicateHandler...")
        self.duplicate_columns = []

        # Find duplicate columns (transpose and check duplicates)
        # Note: Transposing can be expensive for very large datasets,
        # but is accurate. We can optimize or do a simple element-by-element equality check.
        # Let's write a robust, pandas-native column duplication check.
        cols = df.columns
        num_cols = len(cols)
        for i in range(num_cols):
            col_i = cols[i]
            if col_i in self.duplicate_columns:
                continue
            for j in range(i + 1, num_cols):
                col_j = cols[j]
                if col_j in self.duplicate_columns:
                    continue
                # If dtypes match and content matches, mark column as duplicate
                if df[col_i].dtype == df[col_j].dtype:
                    if df[col_i].equals(df[col_j]):
                        logger.info(f"Detected duplicate column: '{col_j}' is identical to '{col_i}'")
                        self.duplicate_columns.append(col_j)

        self.meta_info["duplicate_columns_dropped"] = self.duplicate_columns
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate rows and duplicate columns.

        Args:
            df (pd.DataFrame): DataFrame to clean.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        logger.info("Executing DuplicateHandler transform...")
        current_df = df.copy()

        # 1. Drop duplicate columns identified in fit
        dup_cols_in_df = [c for c in self.duplicate_columns if c in current_df.columns]
        if dup_cols_in_df:
            current_df = current_df.drop(columns=dup_cols_in_df)
            logger.info(f"Dropped duplicate columns: {dup_cols_in_df}")

        # 2. Drop duplicate rows
        initial_rows = len(current_df)
        current_df = current_df.drop_duplicates()
        dropped_rows = initial_rows - len(current_df)
        self.meta_info["duplicate_rows_dropped"] = dropped_rows

        logger.info(f"Duplicate row removal complete. Dropped {dropped_rows} duplicate rows.")
        return current_df
