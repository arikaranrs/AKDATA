"""Feature Selection Module for AKDATA.

Selects features by dropping low-variance or highly redundant (correlated) columns.
"""

import pandas as pd
import numpy as np
from typing import Any, List
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class FeatureSelector(BaseTransformer):
    """Filters out features using variance threshold and correlation thresholding."""

    def __init__(
        self,
        method: str = "correlation",
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.90
    ):
        """Initialize FeatureSelector.

        Args:
            method (str): Filtering method: 'variance', 'correlation', or 'both'.
            variance_threshold (float): Minimum variance required to keep numerical features.
            correlation_threshold (float): Maximum correlation allowed between numerical features.
        """
        super().__init__()
        self.method = method.lower()
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.dropped_columns: List[str] = []

    def fit(self, df: pd.DataFrame, y: Any = None) -> "FeatureSelector":
        """Identify low variance and highly correlated features to drop.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            FeatureSelector: Self instance.
        """
        logger.info(f"Fitting FeatureSelector (method={self.method})...")
        self.dropped_columns = []
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.empty:
            self.is_fitted = True
            return self

        # 1. Variance threshold check
        if self.method in ("variance", "both"):
            variances = numeric_df.var()
            low_var_cols = variances[variances <= self.variance_threshold].index.tolist()
            self.dropped_columns.extend(low_var_cols)
            logger.info(f"Identified {len(low_var_cols)} columns below variance threshold {self.variance_threshold}: {low_var_cols}")

        # 2. Correlation redundancy check
        if self.method in ("correlation", "both"):
            # Compute correlation matrix excluding already dropped columns
            remaining_cols = [c for c in numeric_df.columns if c not in self.dropped_columns]
            if len(remaining_cols) > 1:
                corr_matrix = numeric_df[remaining_cols].corr().abs()
                # Select upper triangle of correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find columns with correlation greater than threshold
                to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
                self.dropped_columns.extend(to_drop)
                logger.info(f"Identified {len(to_drop)} columns exceeding correlation threshold {self.correlation_threshold}: {to_drop}")

        # Make unique list
        self.dropped_columns = list(set(self.dropped_columns))
        self.meta_info["dropped_features"] = self.dropped_columns
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop selected features from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with filtered columns.
        """
        logger.info("Executing FeatureSelector transform...")
        current_df = df.copy()

        cols_to_drop = [c for c in self.dropped_columns if c in current_df.columns]
        if cols_to_drop:
            current_df = current_df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} features during selection.")

        return current_df
