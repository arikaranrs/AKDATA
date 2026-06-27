"""CSV Reader module for AKDATA.

Handles safe CSV loading with automated dialect and delimiter detection.
"""

import csv
import os
import pandas as pd
from typing import Any, Dict
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AKDataIOError

logger = get_logger(__name__)


def load_csv(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame with automatic delimiter detection.

    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional parameters passed to pandas.read_csv.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        AKDataIOError: If file loading fails.
    """
    if not os.path.exists(filepath):
        raise AKDataIOError(f"CSV file not found at: {filepath}")

    logger.info(f"Attempting to load CSV: {filepath}")

    # Detect delimiter
    delimiter = ","
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
            if sample:
                sniffer = csv.Sniffer()
                # Check if it looks like a CSV dialect
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
                logger.info(f"Auto-detected delimiter: '{delimiter}'")
    except Exception as e:
        logger.warning(f"Could not automatically detect CSV delimiter: {e}. Defaulting to ','")

    # Set default sep/delimiter if not specified in kwargs
    if "sep" not in kwargs and "delimiter" not in kwargs:
        kwargs["sep"] = delimiter

    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}")
        return df
    except Exception as e:
        raise AKDataIOError(f"Failed to read CSV '{filepath}': {str(e)}")
