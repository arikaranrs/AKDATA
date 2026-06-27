"""Tests for validation module: target leakage and schema checker."""

import pytest
import pandas as pd
from akdata.validation.leakage import detect_leakage
from akdata.validation.checker import SchemaChecker
from akdata.core.exceptions import ValidationError


def test_detect_leakage():
    """Ensure high target-correlation flags target leakage exception."""
    df = pd.DataFrame({
        "leak": [1, 2, 3, 4, 5],
        "normal": [5, 2, 8, 1, 9]
    })
    y = pd.Series([1, 2, 3, 4, 5])

    # Should raise ValidationError since 'leak' has 100% correlation with y
    with pytest.raises(ValidationError):
        detect_leakage(df, y, threshold=0.95)


def test_schema_checker():
    """Verify schema mismatch throws ValidationError."""
    ref_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    checker = SchemaChecker(ref_df)

    # Valid schema check
    assert checker.validate(ref_df) is True

    # Missing column
    bad_df_1 = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValidationError):
        checker.validate(bad_df_1)

    # Data type mismatch (numerical vs non-numerical)
    bad_df_2 = pd.DataFrame({"a": ["str_val", "str_val2"], "b": ["x", "y"]})
    with pytest.raises(ValidationError):
        checker.validate(bad_df_2)
