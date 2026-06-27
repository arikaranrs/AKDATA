"""AKDATA: Intelligent Data Preprocessing and Validation for Production ML.

An enterprise-grade, modular framework for building clean, leakage-free data pipelines.
"""

from akdata.version import __version__
from akdata import ak
from akdata.core.config import PipelineConfig
from akdata.core.pipeline import Pipeline, BaseTransformer
from akdata.core.orchestrator import Orchestrator, OrchestrationResult
from akdata.core.exceptions import (
    AKDataError,
    PipelineError,
    ValidationError,
    AnalysisError,
    AKDataIOError,
)

__all__ = [
    "__version__",
    "ak",
    "PipelineConfig",
    "Pipeline",
    "BaseTransformer",
    "Orchestrator",
    "OrchestrationResult",
    "AKDataError",
    "PipelineError",
    "ValidationError",
    "AnalysisError",
    "AKDataIOError",
]
