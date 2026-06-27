"""Health Score Module for AKDATA.

Computes a dataset quality health score (0-100) based on weighted metrics
including missingness, duplicates, outliers, and type consistency.
"""

import pandas as pd
from typing import Any, Dict
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AnalysisError
from akdata.utils.constants import (
    WEIGHT_MISSING,
    WEIGHT_DUPLICATES,
    WEIGHT_OUTLIERS,
    WEIGHT_TYPE_CONFLICTS,
    OUTLIER_IQR_FACTOR,
)

logger = get_logger(__name__)


def calculate_health_score(df: pd.DataFrame) -> float:
    """Calculate the data quality score for a DataFrame.

    Health score is a value between 0.0 (worst) and 100.0 (perfect).

    Args:
        df (pd.DataFrame): DataFrame to grade.

    Returns:
        float: Calculated health score out of 100.

    Raises:
        AnalysisError: If calculations fail.
    """
    if df is None or df.empty:
        return 0.0

    logger.info("Calculating dataset health score...")
    try:
        num_rows, num_cols = df.shape
        total_cells = num_rows * num_cols

        # 1. Missing Score
        total_missing = int(df.isnull().sum().sum())
        missing_ratio = total_missing / total_cells if total_cells > 0 else 0.0
        score_missing = (1.0 - missing_ratio) * 100.0

        # 2. Duplicates Score
        duplicate_rows = int(df.duplicated().sum())
        duplicate_ratio = duplicate_rows / num_rows if num_rows > 0 else 0.0
        score_duplicates = (1.0 - duplicate_ratio) * 100.0

        # 3. Outliers Score
        numeric_cols = df.select_dtypes(include=["number"])
        total_numeric_cells = len(numeric_cols) * len(numeric_cols.columns)
        total_outliers = 0

        if not numeric_cols.empty and len(df) > 0:
            for col in numeric_cols.columns:
                series = numeric_cols[col].dropna()
                if len(series) > 0:
                    q25 = series.quantile(0.25)
                    q75 = series.quantile(0.75)
                    iqr = q75 - q25
                    lower = q25 - OUTLIER_IQR_FACTOR * iqr
                    upper = q75 + OUTLIER_IQR_FACTOR * iqr
                    col_outliers = int(((series < lower) | (series > upper)).sum())
                    total_outliers += col_outliers
            outlier_ratio = total_outliers / total_numeric_cells if total_numeric_cells > 0 else 0.0
        else:
            outlier_ratio = 0.0
        score_outliers = (1.0 - outlier_ratio) * 100.0

        # 4. Type conflicts Score
        # We define a type conflict if an 'object' column contains mixed types
        type_conflicts = 0
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            # Check if elements are of different types
            types_set = {type(val) for val in series}
            # Ignore None/nan since we dropped them, but check if there are multiple types left
            if len(types_set) > 1:
                # If we have e.g. int and float, that's fine, but float and string is a conflict
                type_names = {t.__name__ for t in types_set}
                # Filter compatible numerical pairs
                if "str" in type_names and ("int" in type_names or "float" in type_names):
                    type_conflicts += 1

        type_conflict_ratio = type_conflicts / num_cols if num_cols > 0 else 0.0
        score_types = (1.0 - type_conflict_ratio) * 100.0

        # Calculate weighted average
        health_score = (
            (WEIGHT_MISSING * score_missing) +
            (WEIGHT_DUPLICATES * score_duplicates) +
            (WEIGHT_OUTLIERS * score_outliers) +
            (WEIGHT_TYPE_CONFLICTS * score_types)
        )

        return float(round(health_score, 2))
    except Exception as e:
        raise AnalysisError(f"Failed to calculate health score: {str(e)}")
