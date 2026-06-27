"""Tests for Train-Test split wrapper."""

import pytest
import pandas as pd
from akdata.split.train_test import train_test_split_wrapper
from akdata.core.exceptions import ValidationError


def test_train_test_split_wrapper():
    """Verify train-test splitting shapes and compatibility checks."""
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series([0, 1] * 5)

    X_train, X_test, y_train, y_test = train_test_split_wrapper(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    # Incompatible lengths
    with pytest.raises(ValidationError):
        train_test_split_wrapper(X, y.iloc[:5])
