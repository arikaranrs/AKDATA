"""Custom exceptions for AKDATA.

Defines module-specific exceptions for clear debugging and error reporting.
"""


class AKDataError(Exception):
    """Base exception class for all errors in the AKDATA library.

    Provides a clean, custom exception hierarchy for library consumers.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PipelineError(AKDataError):
    """Raised when there is an error in pipeline configuration, fitting, or transforming.

    Indicates issues with step ordering, missing step outputs, or incompatible input data.
    """
    pass


class ValidationError(AKDataError):
    """Raised when dataset verification or target leakage checks fail.

    Indicates that the data does not conform to the expected schema or rules.
    """
    pass


class AnalysisError(AKDataError):
    """Raised when there is an error in calculating EDA stats, profiling, or health scores.

    Indicates problems calculating metrics or analyzing structural properties of the dataset.
    """
    pass


class AKDataIOError(AKDataError):
    """Raised when data loading from CSV, Excel, or JSON fails.

    Indicates missing files, bad file formats, or parser errors.
    """
    pass
