"""Analysis package for AKDATA.

Contains EDA, data profiling, statistics, and dataset health score modules.
"""

from akdata.analysis.eda import get_eda_summary
from akdata.analysis.profiling import profile_dataset
from akdata.analysis.statistics import compute_statistics
from akdata.analysis.health_score import calculate_health_score
from akdata.analysis.recommendations import generate_recommendations

__all__ = [
    "get_eda_summary",
    "profile_dataset",
    "compute_statistics",
    "calculate_health_score",
    "generate_recommendations",
]
