"""Tests for core AKDATA functionalities: config, exceptions, pipeline, and orchestrator."""

import os
import pytest
import pandas as pd
from akdata.core.config import PipelineConfig
from akdata.core.exceptions import AKDataError, PipelineError
from akdata.core.pipeline import Pipeline, BaseTransformer
from akdata.core.orchestrator import Orchestrator


class DummyTransformer(BaseTransformer):
    """Simple transformer for testing pipeline steps."""

    def __init__(self, add_val: int = 1):
        super().__init__()
        self.add_val = add_val

    def fit(self, df: pd.DataFrame, y=None):
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df + self.add_val


def test_pipeline_config():
    """Verify that config parameters load and export correctly."""
    config = PipelineConfig(target_column="target", random_state=123)
    assert config.target_column == "target"
    assert config.random_state == 123

    # Test dictionary export/import
    d = config.to_dict()
    assert d["target_column"] == "target"
    assert d["random_state"] == 123

    config2 = PipelineConfig.from_dict({"target_column": "label", "random_state": 456})
    assert config2.target_column == "label"
    assert config2.random_state == 456


def test_pipeline_execution():
    """Verify sequential fitting and transformation in a custom pipeline."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    pipeline = Pipeline()
    
    # Assert fitting raises error if not fitted
    with pytest.raises(PipelineError):
        pipeline.transform(df)

    # Add steps
    pipeline.add_step("add_one", DummyTransformer(add_val=1))
    pipeline.add_step("add_two", DummyTransformer(add_val=2))

    # Fit and transform
    res = pipeline.fit_transform(df)
    assert res["a"].tolist() == [4, 5, 6]
    assert pipeline.is_fitted is True


def test_orchestrator_initialization():
    """Ensure orchestrator loads with defaults."""
    orch = Orchestrator()
    assert orch.config is not None
    assert isinstance(orch.config, PipelineConfig)


def test_orchestration_result_methods(tmp_path):
    """Test reporting and export helper methods on OrchestrationResult."""
    df_raw = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_clean = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})

    from akdata.core.orchestrator import OrchestrationResult
    from akdata.analysis.eda import get_eda_summary

    eda = get_eda_summary(df_raw)
    res = OrchestrationResult(
        df_clean=df_clean,
        df_raw=df_raw,
        raw_health_score=80.0,
        clean_health_score=100.0,
        eda_summary=eda
    )

    # Test summary string output
    summary_text = res.summary()
    assert "AKDATA PREPROCESSING SUMMARY" in summary_text
    assert "80.00" in summary_text

    # Test HTML export
    html_file = tmp_path / "report.html"
    exported_html_path = res.export_html(path=str(html_file))
    assert os.path.exists(exported_html_path)

    # Test PDF export
    pdf_file = tmp_path / "report.pdf"
    exported_pdf_path = res.export_pdf(path=str(pdf_file))
    assert exported_pdf_path is not None
    assert os.path.exists(exported_pdf_path)


def test_recommendations():
    """Verify recommendation generation rules against a config."""
    import numpy as np
    df = pd.DataFrame({
        "num_col": [1.0, 2.0, np.nan],
        "cat_col": ["X", "Y", "X"]
    })
    
    from akdata.core.config import PipelineConfig
    from akdata.analysis.recommendations import generate_recommendations

    config = PipelineConfig(impute_missing=True, encode_categorical=True)
    recs = generate_recommendations(df, config, {})
    
    assert len(recs) > 0
    # Must flag missing values decision
    has_missing_rec = any("Missing Values" in r["feature"] for r in recs)
    assert has_missing_rec is True


