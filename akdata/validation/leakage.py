"""Target Leakage Detection Module for AKDATA.

Checks for features that are highly predictive of the target column in a way that suggests leakage.
"""

import pandas as pd
from typing import Any, List
from akdata.utils.logger import get_logger
from akdata.core.exceptions import ValidationError

logger = get_logger(__name__)


def detect_leakage(
    df: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.95
) -> List[str]:
    """Detect features that exhibit target leakage based on correlation.

    Args:
        df (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target vector.
        threshold (float): Absolute correlation threshold above which leakage is flagged.

    Returns:
        List[str]: List of leaking feature names.

    Raises:
        ValidationError: If target leakage is detected, listing leaking columns.
    """
    logger.info(f"Checking for target leakage using threshold: {threshold}...")
    
    if df.empty or y is None or len(y) == 0:
        return []

    leaking_features: List[str] = []
    
    # Try to convert target to numeric for correlation calculation
    try:
        y_numeric = pd.to_numeric(y, errors="coerce")
        # If target has many NaNs after conversion, it might be string/categorical
        if y_numeric.isnull().sum() / len(y) > 0.5:
            # Let's map target to categorical integer codes
            y_numeric = pd.Series(pd.Categorical(y).codes, index=y.index)
    except Exception as e:
        logger.warning(f"Could not convert target to numeric for leakage check: {e}")
        return []

    # Iterate through columns in df
    for col in df.columns:
        series = df[col]
        # Only check numeric columns (categorical are already encoded at this stage in pipeline)
        if pd.api.types.is_numeric_dtype(series):
            try:
                # Drop nulls for correlation calc
                valid_mask = series.notnull() & y_numeric.notnull()
                if valid_mask.sum() >= 3:
                    corr = abs(series[valid_mask].corr(y_numeric[valid_mask]))
                    if corr >= threshold:
                        logger.warning(f"Target Leakage Warning: Column '{col}' has a correlation of {corr:.4f} with target.")
                        leaking_features.append(col)
            except Exception as e:
                logger.debug(f"Failed to calculate correlation for leakage on '{col}': {e}")

    if leaking_features:
        raise ValidationError(
            f"Target leakage detected in features: {leaking_features}. "
            f"These features have correlation >= {threshold} with the target. "
            "Please remove or inspect these columns to prevent data leakage."
        )

    logger.info("No target leakage detected.")
    return leaking_features
