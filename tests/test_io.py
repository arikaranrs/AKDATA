"""Tests for AKDATA IO load functions (csv, json, excel)."""

import os
import tempfile
import pytest
import pandas as pd
from akdata.io.csv import load_csv
from akdata.io.json import load_json
from akdata.core.exceptions import AKDataIOError


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df.to_csv(path, index=False, sep=";")
        yield path
    finally:
        os.close(fd)
        os.remove(path)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df.to_json(path, orient="records")
        yield path
    finally:
        os.close(fd)
        os.remove(path)


def test_load_csv_auto_detect(temp_csv_file):
    """Verify delimiter auto-detection works."""
    df = load_csv(temp_csv_file)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["name", "age"]
    assert df["age"].tolist() == [25, 30]


def test_load_csv_missing_file():
    """Ensure missing file raises custom error."""
    with pytest.raises(AKDataIOError):
        load_csv("non_existent_file_path.csv")


def test_load_json(temp_json_file):
    """Ensure JSON is loaded correctly."""
    df = load_json(temp_json_file)
    assert df.shape == (2, 2)
    assert df["name"].tolist() == ["Alice", "Bob"]
