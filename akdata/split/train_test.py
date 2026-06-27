"""Train-Test Split Module for AKDATA.

Implements robust splitting of features and target datasets into train and test sets.
"""

from typing import Tuple, Union, Optional
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split
from akdata.utils.logger import get_logger
from akdata.core.exceptions import ValidationError

logger = get_logger(__name__)


def train_test_split_wrapper(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform train/test split on features X and target y.

    Handles classification stratification constraints gracefully.

    Args:
        X (pd.DataFrame): Features.
        y (Union[pd.Series, pd.DataFrame]): Target vector.
        test_size (float): Portion of dataset to assign to test set (0.0 to 1.0).
        stratify (bool): If True, stratify based on the target column class distribution.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test

    Raises:
        ValidationError: If inputs are invalid or incompatible.
    """
    logger.info("Executing Train-Test split...")

    if X is None or y is None:
        raise ValidationError("X and y must not be None for train-test split.")

    if len(X) != len(y):
        raise ValidationError(f"X (len={len(X)}) and y (len={len(y)}) must have identical lengths.")

    stratify_target = None
    if stratify:
        # Check target class counts for stratification
        # If target has classes with only 1 sample, stratification will fail.
        # Fall back to unstratified split and log a warning.
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        class_counts = y_series.value_counts()
        if (class_counts < 2).any():
            logger.warning(
                "Stratification requested, but some target classes have fewer than 2 samples. "
                "Falling back to non-stratified split."
            )
        else:
            stratify_target = y

    try:
        X_train, X_test, y_train, y_test = sklearn_split(
            X, y,
            test_size=test_size,
            stratify=stratify_target,
            random_state=random_state
        )
        logger.info(f"Split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValidationError(f"Train-Test split failed: {str(e)}")
