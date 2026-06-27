"""Convenience Developer API for AKDATA.

Exposes simple top-level functions for rapid coding: load methods, prepare, etc.
"""

from typing import Any, Optional, Union
import pandas as pd

from akdata.io.csv import load_csv
from akdata.io.excel import load_excel
from akdata.io.json import load_json
from akdata.core.config import PipelineConfig
from akdata.core.orchestrator import Orchestrator, OrchestrationResult


def read_csv(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame with automatic delimiter detection.

    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional arguments passed to pandas.read_csv.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return load_csv(filepath, **kwargs)


def read_excel(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Read an Excel file into a pandas DataFrame.

    Args:
        filepath (str): Path to the Excel file.
        **kwargs: Additional arguments passed to pandas.read_excel.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return load_excel(filepath, **kwargs)


def read_json(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Read a JSON file into a pandas DataFrame.

    Args:
        filepath (str): Path to the JSON file.
        **kwargs: Additional arguments passed to pandas.read_json.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return load_json(filepath, **kwargs)


def prepare(
    df: pd.DataFrame,
    target: Optional[str] = None,
    **kwargs: Any
) -> OrchestrationResult:
    """Prepare a dataset end-to-end for Machine Learning models.

    Coordinates loading, quality inspection, cleaning, imputation, encoding,
    splitting, scaling, and validation.

    Args:
        df (pd.DataFrame): Input DataFrame to prepare.
        target (Optional[str]): Name of the target label column.
        **kwargs: Configuration parameter overrides for PipelineConfig.

    Returns:
        OrchestrationResult: The result containing splits (X_train, y_train, etc.) and pipeline metadata.
    """
    # Build config overrides
    config_dict = kwargs
    if target is not None:
        config_dict["target_column"] = target

    config = PipelineConfig.from_dict(config_dict)
    orchestrator = Orchestrator(config=config)
    return orchestrator.run(df)
