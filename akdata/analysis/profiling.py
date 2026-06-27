"""Data Profiling Module for AKDATA.

Performs deep analysis of column values (cardinality, ranges, distributions, modes).
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AnalysisError
from akdata.utils.constants import OUTLIER_IQR_FACTOR

logger = get_logger(__name__)


def profile_dataset(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Generate detailed profile information for each column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to profile.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping column names to profiling metadata.

    Raises:
        AnalysisError: If the profiling fails.
    """
    if df is None:
        raise AnalysisError("Cannot profile None input.")

    logger.info("Profiling columns...")
    profile: Dict[str, Dict[str, Any]] = {}
    num_rows = len(df)

    try:
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            unique_count = int(series.nunique(dropna=True))
            
            col_profile: Dict[str, Any] = {
                "null_count": null_count,
                "unique_count": unique_count,
                "cardinality_ratio": round(unique_count / num_rows, 4) if num_rows > 0 else 0.0,
            }

            # Numeric columns
            if pd.api.types.is_numeric_dtype(series):
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    col_profile.update({
                        "type": "numeric",
                        "min": float(non_null_series.min()),
                        "max": float(non_null_series.max()),
                        "mean": float(non_null_series.mean()),
                        "median": float(non_null_series.median()),
                        "std": float(non_null_series.std()) if len(non_null_series) > 1 else 0.0,
                        "zeros_count": int((non_null_series == 0).sum()),
                        "negatives_count": int((non_null_series < 0).sum()),
                    })
                    # Outliers detection via IQR
                    q25 = non_null_series.quantile(0.25)
                    q75 = non_null_series.quantile(0.75)
                    iqr = q75 - q25
                    lower_bound = q25 - OUTLIER_IQR_FACTOR * iqr
                    upper_bound = q75 + OUTLIER_IQR_FACTOR * iqr
                    outliers = non_null_series[(non_null_series < lower_bound) | (non_null_series > upper_bound)]
                    col_profile["outliers_count"] = len(outliers)
                else:
                    col_profile.update({
                        "type": "numeric",
                        "min": None, "max": None, "mean": None, "median": None, "std": None,
                        "zeros_count": 0, "negatives_count": 0, "outliers_count": 0
                    })

            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(series):
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    col_profile.update({
                        "type": "datetime",
                        "min": str(non_null_series.min()),
                        "max": str(non_null_series.max()),
                        "range_days": int((non_null_series.max() - non_null_series.min()).days)
                    })
                else:
                    col_profile.update({
                        "type": "datetime",
                        "min": None, "max": None, "range_days": 0
                    })

            # Categorical columns
            else:
                col_profile.update({"type": "categorical"})
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    val_counts = non_null_series.value_counts()
                    mode_val = val_counts.index[0]
                    mode_freq = int(val_counts.iloc[0])
                    col_profile.update({
                        "mode": str(mode_val),
                        "mode_frequency": mode_freq,
                        "mode_percentage": round((mode_freq / len(non_null_series)) * 100, 2)
                    })
                else:
                    col_profile.update({
                        "mode": None, "mode_frequency": 0, "mode_percentage": 0.0
                    })

            profile[col] = col_profile

        return profile
    except Exception as e:
        raise AnalysisError(f"Failed to profile dataset: {str(e)}")
