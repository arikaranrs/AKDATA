"""Data Preprocessing Pipeline for AKDATA.

Implements the fit-transform pipeline pattern to manage sequential preprocessing steps.
"""

from typing import List, Tuple, Any, Dict
import pandas as pd
from akdata.utils.logger import get_logger
from akdata.core.exceptions import PipelineError

logger = get_logger(__name__)


class BaseTransformer:
    """Base class for all data preprocessing transformers in AKDATA.

    Enforces the fit/transform pattern to prevent data leakage.
    """

    def __init__(self):
        self.is_fitted = False
        self.meta_info: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> "BaseTransformer":
        """Fit the transformer on the training data.

        Args:
            df (pd.DataFrame): Training features.
            y (Any, optional): Target column. Defaults to None.

        Returns:
            BaseTransformer: Self instance.
        """
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation on the data.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data.
        """
        return df

    def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit and transform in a single call.

        Args:
            df (pd.DataFrame): Training features.
            y (Any, optional): Target column. Defaults to None.

        Returns:
            pd.DataFrame: Transformed data.
        """
        return self.fit(df, y).transform(df)


class Pipeline:
    """Sequential pipeline execution engine for AKDATA.

    Chains multiple preprocessing steps, executing them in sequence. Tracks
    what transformations were made at each step to maintain auditability.
    """

    def __init__(self, steps: List[Tuple[str, BaseTransformer]] = None):
        """Initialize the pipeline with a list of steps.

        Args:
            steps (List[Tuple[str, BaseTransformer]], optional): Named pipeline steps.
        """
        self.steps = steps if steps is not None else []
        self.meta_info: Dict[str, Any] = {}
        self.is_fitted = False

    def add_step(self, name: str, transformer: BaseTransformer) -> None:
        """Add a step to the end of the pipeline.

        Args:
            name (str): Unique name of the step.
            transformer (BaseTransformer): The transformer instance.

        Raises:
            PipelineError: If a step with the same name already exists.
        """
        if any(step[0] == name for step in self.steps):
            raise PipelineError(f"Step with name '{name}' already exists in pipeline.")
        self.steps.append((name, transformer))

    def fit(self, df: pd.DataFrame, y: Any = None) -> "Pipeline":
        """Fit all steps of the pipeline in sequence.

        Args:
            df (pd.DataFrame): Training data.
            y (Any, optional): Target vector.

        Returns:
            Pipeline: The fitted pipeline instance.
        """
        logger.info("Fitting the preprocessing pipeline...")
        current_df = df.copy()

        for name, transformer in self.steps:
            logger.info(f"Fitting step: {name}")
            transformer.fit(current_df, y)
            current_df = transformer.transform(current_df)
            self.meta_info[name] = transformer.meta_info

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all pipeline steps in sequence to transform the dataset.

        Args:
            df (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The fully transformed dataset.

        Raises:
            PipelineError: If the pipeline is not fitted.
        """
        if not self.is_fitted:
            raise PipelineError("Pipeline is not fitted yet. Call fit() first.")

        logger.info("Transforming dataset using fitted pipeline...")
        current_df = df.copy()

        for name, transformer in self.steps:
            logger.debug(f"Applying transformation step: {name}")
            current_df = transformer.transform(current_df)

        return current_df

    def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit the pipeline and transform the dataset in one pass.

        Args:
            df (pd.DataFrame): Training data.
            y (Any, optional): Target vector.

        Returns:
            pd.DataFrame: Transformed training dataset.
        """
        self.fit(df, y)
        return self.transform(df)
