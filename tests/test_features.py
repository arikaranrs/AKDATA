"""Tests for features engineering and selection modules."""

import pytest
import pandas as pd
from akdata.features.engineering import FeatureEngineer
from akdata.features.selection import FeatureSelector


def test_feature_engineer():
    """Verify datetime feature extraction and interactions creation."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-05-10", "2023-06-15"]),
        "val1": [2, 3],
        "val2": [4, 5]
    })
    engineer = FeatureEngineer(create_interactions=True)
    res = engineer.fit_transform(df)

    assert "date_year" in res.columns
    assert "date_month" in res.columns
    assert "date_day" in res.columns
    assert "val1_x_val2" in res.columns
    assert res["val1_x_val2"].tolist() == [8, 15]


def test_feature_selector():
    """Ensure low-variance and highly correlated features are dropped."""
    df = pd.DataFrame({
        "constant": [1, 1, 1],  # zero variance
        "feat1": [1, 2, 3],
        "feat2": [2, 4, 6]      # perfectly correlated with feat1
    })

    selector = FeatureSelector(method="both", variance_threshold=0.01, correlation_threshold=0.95)
    res = selector.fit_transform(df)

    # constant column should be dropped (low variance)
    assert "constant" not in res.columns
    # one of feat1/feat2 should be dropped due to high correlation
    assert ("feat1" not in res.columns) or ("feat2" not in res.columns)
