"""Dataset Schema Validation Module for AKDATA.

Checks if a dataset conforms to a reference schema of columns and types.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from akdata.utils.logger import get_logger
from akdata.core.exceptions import ValidationError

logger = get_logger(__name__)


class SchemaChecker:
    """Validator to ensure a DataFrame conforms to a saved schema."""

    def __init__(self, reference_df: Optional[pd.DataFrame] = None):
        """Initialize SchemaChecker.

        Args:
            reference_df (Optional[pd.DataFrame]): If provided, infers the reference schema from it.
        """
        self.expected_dtypes: Dict[str, str] = {}
        self.expected_columns: List[str] = []
        if reference_df is not None:
            self.fit(reference_df)

    def fit(self, df: pd.DataFrame) -> "SchemaChecker":
        """Save the column names and data types of the reference DataFrame as the schema.

        Args:
            df (pd.DataFrame): Reference DataFrame.

        Returns:
            SchemaChecker: Self instance.
        """
        logger.info("Recording schema from reference DataFrame...")
        self.expected_columns = df.columns.tolist()
        self.expected_dtypes = {col: str(df[col].dtype) for col in df.columns}
        return self

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate if a DataFrame conforms to the recorded schema.

        Args:
            df (pd.DataFrame): DataFrame to check.

        Returns:
            bool: True if validation passes.

        Raises:
            ValidationError: If columns are missing or data types mismatch.
        """
        logger.info("Validating DataFrame schema...")
        
        # 1. Check for missing columns
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            raise ValidationError(
                f"Schema Validation Failed: Missing expected columns: {missing_cols}. "
                f"Expected columns: {self.expected_columns}"
            )

        # 2. Check for data type mismatches
        dtype_mismatches = []
        for col in self.expected_columns:
            actual_type = str(df[col].dtype)
            expected_type = self.expected_dtypes[col]
            
            # Allow some tolerance (e.g. float64 and int64 are compatible numerical types,
            # category and object might also be acceptable depending on constraints)
            if actual_type != expected_type:
                # Flag if one is numeric and the other is object/string
                is_actual_num = "int" in actual_type or "float" in actual_type
                is_expected_num = "int" in expected_type or "float" in expected_type
                if is_actual_num != is_expected_num:
                    dtype_mismatches.append(f"Column '{col}' expected '{expected_type}', found '{actual_type}'")

        if dtype_mismatches:
            raise ValidationError(
                f"Schema Validation Failed: Data type conflicts detected: {dtype_mismatches}"
            )

        logger.info("DataFrame schema validation successful.")
        return True
