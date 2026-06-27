"""Outliers Preprocessor Module for AKDATA.

Identifies and handles outliers in numerical columns using IQR or Z-score.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List
from akdata.utils.logger import get_logger
from akdata.core.pipeline import BaseTransformer

logger = get_logger(__name__)


class OutlierHandler(BaseTransformer):
    """Detects and handles outliers in numerical columns.

    Supports clipping (capping/flooring) and dropping strategies.
    Uses IQR or Z-score limits computed on training data.
    """

    def __init__(self, method: str = "iqr", threshold: float = 1.5, action: str = "clip"):
        """Initialize the OutlierHandler.

        Args:
            method (str): Detection method: 'iqr' or 'zscore'.
            threshold (float): Factor for IQR (default 1.5) or standard deviations for Z-score (default 3.0).
            action (str): Response action: 'clip' (cap/floor values) or 'drop' (remove rows).
        """
        super().__init__()
        self.method = method.lower()
        self.threshold = threshold
        self.action = action.lower()
        self.bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "OutlierHandler":
        """Compute outlier thresholds for numerical columns.

        Args:
            df (pd.DataFrame): Training DataFrame.
            y (Any, optional): Target vector.

        Returns:
            OutlierHandler: Self instance.
        """
        logger.info(f"Fitting OutlierHandler (method={self.method}, threshold={self.threshold}, action={self.action})...")
        self.bounds = {}
        numeric_cols = df.select_dtypes(include=["number"])

        for col in numeric_cols.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if self.method == "iqr":
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - (self.threshold * iqr)
                upper = q75 + (self.threshold * iqr)
            elif self.method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    lower = mean
                    upper = mean
                else:
                    lower = mean - (self.threshold * std)
                    upper = mean + (self.threshold * std)
            else:
                logger.warning(f"Unknown outlier method '{self.method}'. Defaulting to IQR.")
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - (1.5 * iqr)
                upper = q75 + (1.5 * iqr)

            self.bounds[col] = (float(lower), float(upper))

        self.meta_info["outlier_bounds"] = self.bounds
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier treatment to the dataset.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Executing OutlierHandler transform...")
        current_df = df.copy()

        # If action is drop, keep track of rows to drop
        rows_to_drop = pd.Series(False, index=current_df.index)
        outliers_detected = {}

        for col, (lower, upper) in self.bounds.items():
            if col in current_df.columns:
                series = current_df[col]
                is_outlier = (series < lower) | (series > upper)
                outlier_count = int(is_outlier.sum())
                outliers_detected[col] = outlier_count

                if outlier_count > 0:
                    if self.action == "clip":
                        # Clip values to bounds
                        current_df[col] = current_df[col].clip(lower=lower, upper=upper)
                        logger.debug(f"Clipped {outlier_count} outliers in column '{col}' to range [{lower:.2f}, {upper:.2f}].")
                    elif self.action == "drop":
                        rows_to_drop = rows_to_drop | is_outlier

        if self.action == "drop" and rows_to_drop.any():
            initial_len = len(current_df)
            current_df = current_df[~rows_to_drop]
            dropped_count = initial_len - len(current_df)
            logger.info(f"Dropped {dropped_count} rows containing outliers.")
            self.meta_info["outliers_rows_dropped"] = dropped_count
        else:
            self.meta_info["outliers_rows_dropped"] = 0

        self.meta_info["outliers_counts"] = outliers_detected
        return current_df
