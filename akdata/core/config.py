"""Configuration management for AKDATA pipelines.

Provides configuration settings for the data orchestrator and pipeline execution.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from akdata.utils.constants import DEFAULT_RANDOM_STATE


@dataclass
class PipelineConfig:
    """Configuration class for the AKDATA Pipeline and Orchestrator.

    Holds settings for preprocessing, validation, splitting, and reporting.
    """

    # Core settings
    target_column: Optional[str] = None
    random_state: int = DEFAULT_RANDOM_STATE

    # Preprocessing settings
    clean_columns: bool = True
    impute_missing: bool = True
    numeric_imputation_strategy: str = "median"  # 'mean', 'median', 'mode', 'constant'
    categorical_imputation_strategy: str = "mode"  # 'mode', 'constant'
    impute_constant_value: Any = None
    
    remove_duplicates: bool = True
    
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # 'iqr', 'zscore'
    outlier_threshold: float = 1.5  # IQR factor or Z-score threshold
    
    auto_cast_dtypes: bool = True
    
    encode_categorical: bool = True
    encoding_strategy: str = "onehot"  # 'onehot', 'label'
    
    scale_numeric: bool = True
    scaling_strategy: str = "standard"  # 'standard', 'minmax'
    
    math_transformations: Optional[Dict[str, str]] = field(default_factory=dict)  # col -> 'log', 'sqrt'

    # Features settings
    feature_engineering: bool = True
    feature_selection: bool = True
    feature_selection_method: str = "correlation"  # 'correlation', 'variance'
    correlation_threshold: float = 0.90
    variance_threshold: float = 0.01

    # Split settings
    split_data: bool = True
    test_size: float = 0.2
    stratify: bool = False

    # Validation settings
    check_leakage: bool = True
    leakage_threshold: float = 0.95

    # Reporting settings
    generate_html_report: bool = True
    generate_pdf_report: bool = True
    report_dir: str = "reports"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the config.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create a PipelineConfig instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary of config keys and values.

        Returns:
            PipelineConfig: A validated config instance.
        """
        # Filter keys to match dataclass fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
