"""Tests for HTML, PDF, and summary reporting modules."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from akdata.reports.summary import generate_summary_report
from akdata.reports.html import HTMLReporter
from akdata.reports.pdf import PDFReporter
from akdata.analysis.eda import get_eda_summary


@pytest.fixture
def sample_data():
    """Create sample raw/clean dataframes and summary."""
    df_raw = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "y", "z"]})
    df_clean = pd.DataFrame({"a": [1.0, 2.0, 1.5], "b": ["x", "y", "z"]})
    eda_summary = get_eda_summary(df_raw)
    return df_raw, df_clean, eda_summary


def test_generate_summary_report(sample_data):
    """Ensure summary report is a non-empty string containing keys."""
    df_raw, df_clean, eda_summary = sample_data
    report = generate_summary_report(df_raw, df_clean, 80.0, 95.0, eda_summary)
    assert isinstance(report, str)
    assert "AKDATA PREPROCESSING SUMMARY" in report
    assert "80.00" in report
    assert "95.00" in report


def test_html_reporter(sample_data):
    """Verify HTML report generation."""
    df_raw, df_clean, eda_summary = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = HTMLReporter(output_dir=tmpdir)
        path = reporter.generate(df_raw, df_clean, 80.0, 95.0, eda_summary)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "AKDATA" in content
            assert "html" in path


def test_pdf_reporter(sample_data):
    """Verify PDF report generation fallback or completion."""
    df_raw, df_clean, eda_summary = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = PDFReporter(output_dir=tmpdir)
        path = reporter.generate(df_raw, df_clean, 80.0, 95.0, eda_summary)
        assert path is not None
        assert os.path.exists(path)
