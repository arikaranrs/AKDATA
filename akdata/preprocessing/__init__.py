"""Preprocessing package for AKDATA.

Contains data cleaning, missing value imputation, duplicates handling,
outliers handling, datatype casting, categorical encoding, scaling,
and mathematical transformation steps.
"""

from akdata.preprocessing.cleaning import DataCleaner
from akdata.preprocessing.missing import MissingValueImputer
from akdata.preprocessing.duplicates import DuplicateHandler
from akdata.preprocessing.outliers import OutlierHandler
from akdata.preprocessing.datatype import DataTypeCaster
from akdata.preprocessing.encoding import CategoricalEncoder
from akdata.preprocessing.scaling import NumericScaler
from akdata.preprocessing.transformers import MathTransformer

__all__ = [
    "DataCleaner",
    "MissingValueImputer",
    "DuplicateHandler",
    "OutlierHandler",
    "DataTypeCaster",
    "CategoricalEncoder",
    "NumericScaler",
    "MathTransformer",
]
