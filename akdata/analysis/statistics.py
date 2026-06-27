"""Statistics Module for AKDATA.

Computes skewness, kurtosis, and correlation matrices for dataset columns.
"""

from typing import Dict, Any
import pandas as pd
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AnalysisError

logger = get_logger(__name__)


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute mathematical statistics for numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - skewness: Dict mapping column name to its skewness value.
            - kurtosis: Dict mapping column name to its kurtosis value.
            - correlation_matrix: The correlation matrix as a nested dict (col -> other_col -> value).

    Raises:
        AnalysisError: If calculations fail.
    """
    if df is None:
        raise AnalysisError("Cannot compute statistics on None input.")

    logger.info("Computing mathematical statistics...")
    numeric_cols = df.select_dtypes(include=["number"])

    try:
        stats: Dict[str, Any] = {
            "skewness": {},
            "kurtosis": {},
            "correlation_matrix": {}
        }

        # Compute skewness & kurtosis for numeric columns
        for col in numeric_cols.columns:
            series = numeric_cols[col].dropna()
            if len(series) > 1:
                stats["skewness"][col] = float(series.skew())
                stats["kurtosis"][col] = float(series.kurtosis())
            else:
                stats["skewness"][col] = 0.0
                stats["kurtosis"][col] = 0.0

        # Compute correlation matrix
        if not numeric_cols.empty:
            corr_df = numeric_cols.corr(method="pearson")
            # Convert correlation DataFrame to a dictionary
            stats["correlation_matrix"] = corr_df.to_dict()
        else:
            stats["correlation_matrix"] = {}

        return stats
    except Exception as e:
        raise AnalysisError(f"Failed to calculate statistics: {str(e)}")
