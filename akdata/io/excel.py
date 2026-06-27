"""Excel Reader module for AKDATA.

Handles loading data from Excel spread-sheets safely.
"""

import os
import pandas as pd
from typing import Any
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AKDataIOError

logger = get_logger(__name__)


def load_excel(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Read an Excel file (.xlsx, .xls) into a pandas DataFrame.

    Args:
        filepath (str): Path to the Excel file.
        **kwargs: Additional parameters passed to pandas.read_excel.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        AKDataIOError: If file loading fails.
    """
    if not os.path.exists(filepath):
        raise AKDataIOError(f"Excel file not found at: {filepath}")

    logger.info(f"Attempting to load Excel: {filepath}")

    try:
        # Check if openpyxl is needed
        df = pd.read_excel(filepath, **kwargs)
        logger.info(f"Successfully loaded Excel. Shape: {df.shape}")
        return df
    except Exception as e:
        raise AKDataIOError(f"Failed to read Excel file '{filepath}': {str(e)}")
