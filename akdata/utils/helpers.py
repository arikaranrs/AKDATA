"""Helper functions for the AKDATA library.

Contains miscellaneous utility functions for file handling and validations.
"""

import os
from typing import Any, Union
import pandas as pd


def ensure_directory(path: str) -> None:
    """Ensure that the directory for a given file path exists.

    Args:
        path (str): The file or directory path.
    """
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def is_numeric_dtype(series: pd.Series) -> bool:
    """Check if a pandas Series is numeric.

    Args:
        series (pd.Series): The series to check.

    Returns:
        bool: True if the series is numeric, False otherwise.
    """
    return pd.api.types.is_numeric_dtype(series)


def is_datetime_dtype(series: pd.Series) -> bool:
    """Check if a pandas Series represents datetime.

    Args:
        series (pd.Series): The series to check.

    Returns:
        bool: True if the series is datetime, False otherwise.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Try converting a sample to check if it's parseable (optional, but keep it robust)
    return False
