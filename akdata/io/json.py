"""JSON Reader module for AKDATA.

Handles loading data from JSON files safely.
"""

import os
import pandas as pd
from typing import Any
from akdata.utils.logger import get_logger
from akdata.core.exceptions import AKDataIOError

logger = get_logger(__name__)


def load_json(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Read a JSON file into a pandas DataFrame.

    Args:
        filepath (str): Path to the JSON file.
        **kwargs: Additional parameters passed to pandas.read_json.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        AKDataIOError: If file loading fails.
    """
    if not os.path.exists(filepath):
        raise AKDataIOError(f"JSON file not found at: {filepath}")

    logger.info(f"Attempting to load JSON: {filepath}")

    try:
        # If orientation is not specified, try to let pandas figure it out
        df = pd.read_json(filepath, **kwargs)
        logger.info(f"Successfully loaded JSON. Shape: {df.shape}")
        return df
    except Exception as e:
        raise AKDataIOError(f"Failed to read JSON file '{filepath}': {str(e)}")
