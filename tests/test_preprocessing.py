"""Tests for all preprocessing transformers in AKDATA."""

import pytest
import pandas as pd
import numpy as np
from akdata.preprocessing.cleaning import DataCleaner
from akdata.preprocessing.missing import MissingValueImputer
from akdata.preprocessing.duplicates import DuplicateHandler
from akdata.preprocessing.outliers import OutlierHandler
from akdata.preprocessing.datatype import DataTypeCaster
from akdata.preprocessing.encoding import CategoricalEncoder
from akdata.preprocessing.scaling import NumericScaler
from akdata.preprocessing.transformers import MathTransformer


def test_data_cleaner():
    """Verify column standardizing and stripping whitespace."""
    df = pd.DataFrame({
        " First-Name ": [" Alice ", " Bob "],
        "Empty-Col": [np.nan, np.nan]
    })
    cleaner = DataCleaner()
    res = cleaner.fit_transform(df)

    assert "first_name" in res.columns
    assert "empty_col" not in res.columns
    assert res["first_name"].tolist() == ["Alice", "Bob"]


def test_missing_value_imputer():
    """Verify that numerical and categorical values are imputed correctly."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan],
        "cat": ["A", "B", np.nan]
    })
    imputer = MissingValueImputer(numeric_strategy="mean", categorical_strategy="mode")
    res = imputer.fit_transform(df)

    assert res["num"].tolist() == [1.0, 2.0, 1.5]
    assert res["cat"].tolist() == ["A", "B", "A"]  # A is first mode alphabetically/frequent


def test_duplicate_handler():
    """Ensure duplicate columns and rows are dropped."""
    df = pd.DataFrame({
        "a": [1, 2, 1],
        "b": [1, 2, 1],  # Duplicate of a
        "c": [4, 5, 4]
    })
    handler = DuplicateHandler()
    res = handler.fit_transform(df)

    # Column b should be dropped as it is identical to a
    assert "b" not in res.columns
    # Row index 2 is duplicate, should be dropped
    assert len(res) == 2


def test_outlier_handler():
    """Verify that numeric outliers are clipped correctly."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})  # 100 is an outlier
    handler = OutlierHandler(method="iqr", threshold=1.5, action="clip")
    res = handler.fit_transform(df)
    # The value 100.0 should be clipped to the upper bound
    assert res["a"].max() < 100.0


def test_datatype_caster():
    """Ensure string columns are cast to semantic categories."""
    df = pd.DataFrame({
        "a": ["1.2", "2.3", "3.4"],
        "b": ["2023-01-01", "2023-01-02", "2023-01-03"]
    })
    caster = DataTypeCaster()
    res = caster.fit_transform(df)
    assert pd.api.types.is_numeric_dtype(res["a"])
    assert pd.api.types.is_datetime64_any_dtype(res["b"])


def test_categorical_encoder_onehot():
    """Ensure consistent onehot columns mapping."""
    df = pd.DataFrame({"cat": ["A", "B", "A"]})
    encoder = CategoricalEncoder(strategy="onehot")
    res = encoder.fit_transform(df)

    assert "cat_A" in res.columns
    assert "cat_B" in res.columns
    assert res["cat_A"].tolist() == [1, 0, 1]


def test_numeric_scaler():
    """Ensure standard and minmax scaling work correctly."""
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
    scaler = NumericScaler(strategy="minmax")
    res = scaler.fit_transform(df)

    assert res["a"].tolist() == [0.0, 0.5, 1.0]


def test_math_transformer():
    """Ensure math methods like log and sqrt run safely."""
    df = pd.DataFrame({"a": [1.0, 4.0, 9.0]})
    transformer = MathTransformer(transformations={"a": "sqrt"})
    res = transformer.fit_transform(df)

    assert res["a"].tolist() == [1.0, 2.0, 3.0]
