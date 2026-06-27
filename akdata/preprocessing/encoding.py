"""Categorical Encoder Preprocessor Module for AKDATA.

Performs robust one-hot encoding or label encoding on categorical columns.
Ensures consistent column schema between training and testing splits.
"""

import pandas as pd
from typing import Any, Dict, List
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class CategoricalEncoder(BaseTransformer):
    """Encodes categorical columns using One-Hot or Label encoding.

    Guarantees that train and test outputs have identical schemas.
    """

    def __init__(self, strategy: str = "onehot"):
        """Initialize the CategoricalEncoder.

        Args:
            strategy (str): Encoding strategy: 'onehot' or 'label'.
        """
        super().__init__()
        self.strategy = strategy.lower()
        self.categories: Dict[str, List[Any]] = {}
        self.label_mappings: Dict[str, Dict[Any, int]] = {}
        self.encoded_column_names: List[str] = []

    def fit(self, df: pd.DataFrame, y: Any = None) -> "CategoricalEncoder":
        """Learn categories from training data.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            CategoricalEncoder: Self instance.
        """
        logger.info(f"Fitting CategoricalEncoder (strategy={self.strategy})...")
        self.categories = {}
        self.label_mappings = {}
        self.encoded_column_names = []

        # Find categorical columns (object, category, string)
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns

        for col in cat_cols:
            # Sort unique values for determinism
            series = df[col].astype(str)
            unique_vals = sorted(series.unique().tolist())
            self.categories[col] = unique_vals

            if self.strategy == "label":
                # Create dictionary mapping category string -> int index
                self.label_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

        self.meta_info["encoded_categories"] = self.categories
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to the columns of the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to encode.

        Returns:
            pd.DataFrame: DataFrame with encoded columns.
        """
        logger.info("Executing CategoricalEncoder transform...")
        current_df = df.copy()

        for col, cats in self.categories.items():
            if col not in current_df.columns:
                continue

            if self.strategy == "onehot":
                # 1. Convert series to string type
                series = current_df[col].astype(str)
                
                # 2. For each category learned in fit, create a dummy column
                for cat in cats:
                    dummy_col_name = f"{col}_{cat}"
                    current_df[dummy_col_name] = (series == cat).astype(int)
                
                # 3. Drop original column
                current_df = current_df.drop(columns=[col])

            elif self.strategy == "label":
                # Convert series to string type
                series = current_df[col].astype(str)
                mapping = self.label_mappings[col]
                
                # Map categories, default to -1 for unseen categories
                current_df[col] = series.map(mapping).fillna(-1).astype(int)

        logger.info("Categorical encoding complete.")
        return current_df
