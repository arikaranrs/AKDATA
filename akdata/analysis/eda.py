"""Exploratory Data Analysis (EDA) Module for AKDATA.

Computes shape, memory, types, and generic column summaries for a DataFrame.
"""

from typing import Any, Dict, List
import pandas as pd
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AnalysisError

logger = get_logger(__name__)


def get_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic structural summary of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        Dict[str, Any]: Summary dictionary containing:
            - num_rows: Number of rows.
            - num_cols: Number of columns.
            - memory_usage_bytes: Total memory usage in bytes.
            - memory_usage_mb: Memory usage in Megabytes.
            - columns: List of dictionaries with column details (name, dtype, non_null_count, null_count, null_percentage).
            - dtypes_count: Count of columns by general data type.

    Raises:
        AnalysisError: If the DataFrame is empty or calculation fails.
    """
    if df is None:
        raise AnalysisError("Cannot run EDA on None input.")
    
    logger.info("Computing EDA summary...")
    try:
        num_rows, num_cols = df.shape
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())
        memory_usage_mb = round(memory_usage_bytes / (1024 * 1024), 4)

        columns_info: List[Dict[str, Any]] = []
        dtypes_count: Dict[str, int] = {"numeric": 0, "categorical": 0, "datetime": 0, "other": 0}

        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            non_null_count = num_rows - null_count
            null_percentage = round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0.0
            dtype_str = str(series.dtype)

            # Categorize dtypes
            if pd.api.types.is_numeric_dtype(series):
                dtypes_count["numeric"] += 1
                col_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                dtypes_count["datetime"] += 1
                col_type = "datetime"
            elif pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
                dtypes_count["categorical"] += 1
                col_type = "categorical"
            else:
                dtypes_count["other"] += 1
                col_type = "other"

            columns_info.append({
                "name": col,
                "dtype": dtype_str,
                "type_category": col_type,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "null_percentage": null_percentage
            })

        summary = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "memory_usage_bytes": memory_usage_bytes,
            "memory_usage_mb": memory_usage_mb,
            "columns": columns_info,
            "dtypes_count": dtypes_count
        }
        return summary
    except Exception as e:
        raise AnalysisError(f"Failed to generate EDA summary: {str(e)}")
