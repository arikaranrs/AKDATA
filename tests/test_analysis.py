"""Tests for analysis modules: EDA, profiling, statistics, and health score."""

import pytest
import pandas as pd
import numpy as np
from akdata.analysis.eda import get_eda_summary
from akdata.analysis.profiling import profile_dataset
from akdata.analysis.statistics import compute_statistics
from akdata.analysis.health_score import calculate_health_score
from akdata.core.exceptions import AnalysisError


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe with numerical, categorical, and missing values."""
    return pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, np.nan],
        "cat": ["A", "B", "A", np.nan, "B"],
        "date": pd.date_range("2023-01-01", periods=5)
    })


def test_get_eda_summary(sample_dataframe):
    """Verify basic EDA info extraction."""
    summary = get_eda_summary(sample_dataframe)
    assert summary["num_rows"] == 5
    assert summary["num_cols"] == 3
    assert len(summary["columns"]) == 3
    assert summary["dtypes_count"]["numeric"] == 1
    assert summary["dtypes_count"]["categorical"] == 1


def test_profile_dataset(sample_dataframe):
    """Ensure data profiling captures statistical aggregates and counts."""
    profile = profile_dataset(sample_dataframe)
    assert "num" in profile
    assert profile["num"]["null_count"] == 1
    assert profile["num"]["min"] == 1.0
    assert profile["num"]["mean"] == 2.5
    assert profile["cat"]["type"] == "categorical"
    assert profile["cat"]["unique_count"] == 2


def test_compute_statistics(sample_dataframe):
    """Ensure math stats like correlation and skewness are calculated."""
    stats = compute_statistics(sample_dataframe)
    assert "skewness" in stats
    assert "kurtosis" in stats
    assert "correlation_matrix" in stats
    # Check numeric columns
    assert "num" in stats["skewness"]


def test_calculate_health_score(sample_dataframe):
    """Verify that health score computes correctly and changes logically with dirtier data."""
    clean_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dirty_df = pd.DataFrame({
        "a": [1, np.nan, 3, 1, 1],
        "b": [4, 5, np.nan, 4, 4]
    })

    clean_score = calculate_health_score(clean_df)
    dirty_score = calculate_health_score(dirty_df)

    assert clean_score == 100.0
    assert dirty_score < 100.0
    assert dirty_score >= 0.0
