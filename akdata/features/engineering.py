"""Feature Engineering Module for AKDATA.

Generates new features like datetime components and simple numerical interactions.
"""

import pandas as pd
from typing import Any, List, Tuple
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class FeatureEngineer(BaseTransformer):
    """Engineers new features from existing columns.

    Extracts parts of date-times and calculates basic numerical column products.
    """

    def __init__(self, create_interactions: bool = False):
        """Initialize FeatureEngineer.

        Args:
            create_interactions (bool): If True, multiplies numeric features.
        """
        super().__init__()
        self.create_interactions = create_interactions
        self.datetime_cols: List[str] = []
        self.interaction_pairs: List[Tuple[str, str]] = []

    def fit(self, df: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        """Determine what new features can be engineered.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            FeatureEngineer: Self instance.
        """
        logger.info("Fitting FeatureEngineer...")
        self.datetime_cols = []
        self.interaction_pairs = []

        # 1. Identify datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_cols.append(col)

        # 2. Identify numeric pairs for interaction if requested
        if self.create_interactions:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            # If there are a reasonable number of numerical columns, create interactions
            if 1 < len(numeric_cols) < 10:
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        self.interaction_pairs.append((numeric_cols[i], numeric_cols[j]))

        self.meta_info["datetime_features"] = self.datetime_cols
        self.meta_info["interaction_pairs"] = self.interaction_pairs
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        logger.info("Executing FeatureEngineer transform...")
        current_df = df.copy()

        # 1. Datetime feature extraction
        for col in self.datetime_cols:
            if col in current_df.columns:
                series = pd.to_datetime(current_df[col], errors="coerce")
                current_df[f"{col}_year"] = series.dt.year.fillna(0).astype(int)
                current_df[f"{col}_month"] = series.dt.month.fillna(0).astype(int)
                current_df[f"{col}_day"] = series.dt.day.fillna(0).astype(int)
                current_df[f"{col}_dayofweek"] = series.dt.dayofweek.fillna(0).astype(int)
                # Drop original datetime column to avoid errors in numeric models
                current_df = current_df.drop(columns=[col])
                logger.info(f"Extracted datetime components from '{col}' and dropped original.")

        # 2. Numerical interactions
        for col1, col2 in self.interaction_pairs:
            if col1 in current_df.columns and col2 in current_df.columns:
                interaction_name = f"{col1}_x_{col2}"
                current_df[interaction_name] = current_df[col1] * current_df[col2]
                logger.debug(f"Created interaction feature: '{interaction_name}'.")

        return current_df
