"""Orchestrator Module for AKDATA.

Coordinates the end-to-end data processing workflow:
loading, analysis, cleaning, validation, preprocessing, feature engineering,
splitting, scaling, validation, and reporting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from akdata.utils.logger import get_logger
from akdata.core.config import PipelineConfig
from akdata.core.exceptions import AKDataError, ValidationError
from akdata.core.pipeline import Pipeline

# Dynamic imports are handled internally to avoid circular references
logger = get_logger(__name__)


import os


class OrchestrationResult:
    """Encapsulates the output of the AKDATA Orchestrator execution."""

    def __init__(
        self,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[Union[pd.Series, pd.DataFrame]] = None,
        y_test: Optional[Union[pd.Series, pd.DataFrame]] = None,
        df_clean: Optional[pd.DataFrame] = None,
        df_raw: Optional[pd.DataFrame] = None,
        pipeline: Optional[Pipeline] = None,
        config: Optional[PipelineConfig] = None,
        raw_health_score: float = 0.0,
        clean_health_score: float = 0.0,
        report_paths: Optional[Dict[str, str]] = None,
        eda_summary: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[Dict[str, str]]] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_clean = df_clean
        self.df_raw = df_raw
        self.pipeline = pipeline
        self.config = config
        self.raw_health_score = raw_health_score
        self.clean_health_score = clean_health_score
        self.report_paths = report_paths or {}
        self.eda_summary = eda_summary or {}
        self.recommendations = recommendations or []

    def summary(self) -> str:
        """Print and return a text-based summary of dataset profiling and changes.

        Returns:
            str: The summary report text.
        """
        from akdata.reports.summary import generate_summary_report

        df_eval = self.df_clean
        target = self.config.target_column if self.config else None
        if df_eval is None and self.X_train is not None:
            df_eval = self.X_train.copy()
            if self.y_train is not None and target is not None:
                df_eval[target] = self.y_train

        df_raw_eval = self.df_raw if self.df_raw is not None else df_eval

        report = generate_summary_report(
            df_raw_eval,
            df_eval,
            self.raw_health_score,
            self.clean_health_score,
            self.eda_summary
        )
        print(report)
        return report

    def export_html(self, path: Optional[str] = None) -> str:
        """Export the interactive HTML profile dashboard.

        Args:
            path (Optional[str]): Explicit file path to save the HTML.
                If None, saves in the config-specified report directory.

        Returns:
            str: Path to the generated HTML file.
        """
        from akdata.reports.html import HTMLReporter

        output_dir = self.config.report_dir if self.config else "reports"
        if path:
            output_dir = os.path.dirname(path) or "."

        reporter = HTMLReporter(output_dir=output_dir)

        df_eval = self.df_clean
        target = self.config.target_column if self.config else None
        if df_eval is None and self.X_train is not None:
            df_eval = self.X_train.copy()
            if self.y_train is not None and target is not None:
                df_eval[target] = self.y_train

        df_raw_eval = self.df_raw if self.df_raw is not None else df_eval

        html_path = reporter.generate(
            df_raw=df_raw_eval,
            df_clean=df_eval,
            raw_health_score=self.raw_health_score,
            clean_health_score=self.clean_health_score,
            eda_summary=self.eda_summary,
            pipeline_meta=self.pipeline.meta_info if self.pipeline else None
        )

        if path and os.path.basename(path):
            new_path = path
            if os.path.exists(html_path):
                os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(html_path, new_path)
                html_path = new_path

        self.report_paths["html"] = html_path
        return html_path

    def export_pdf(self, path: Optional[str] = None) -> Optional[str]:
        """Export the PDF summary profile report.

        Args:
            path (Optional[str]): Explicit file path to save the PDF.
                If None, saves in the config-specified report directory.

        Returns:
            Optional[str]: Path to the generated PDF file, or None if generation failed.
        """
        from akdata.reports.pdf import PDFReporter

        output_dir = self.config.report_dir if self.config else "reports"
        if path:
            output_dir = os.path.dirname(path) or "."

        reporter = PDFReporter(output_dir=output_dir)

        df_eval = self.df_clean
        target = self.config.target_column if self.config else None
        if df_eval is None and self.X_train is not None:
            df_eval = self.X_train.copy()
            if self.y_train is not None and target is not None:
                df_eval[target] = self.y_train

        df_raw_eval = self.df_raw if self.df_raw is not None else df_eval

        pdf_path = reporter.generate(
            df_raw=df_raw_eval,
            df_clean=df_eval,
            raw_health_score=self.raw_health_score,
            clean_health_score=self.clean_health_score,
            eda_summary=self.eda_summary,
            pipeline_meta=self.pipeline.meta_info if self.pipeline else None
        )

        if path and pdf_path and os.path.basename(path):
            new_path = path
            if os.path.exists(pdf_path):
                os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(pdf_path, new_path)
                pdf_path = new_path

        if pdf_path:
            self.report_paths["pdf"] = pdf_path
        return pdf_path


class Orchestrator:
    """Main Orchestrator class for AKDATA.

    Coordinates the execution of the entire data science preprocessing lifecycle.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the Orchestrator with a PipelineConfig.

        Args:
            config (Optional[PipelineConfig]): The configuration to use.
                If None, a default PipelineConfig is created.
        """
        self.config = config if config is not None else PipelineConfig()
        self.pipeline = Pipeline()
        logger.info("AKDATA Orchestrator initialized.")

    def run(self, data: Union[str, pd.DataFrame]) -> OrchestrationResult:
        """Run the end-to-end data orchestration lifecycle.

        Args:
            data (Union[str, pd.DataFrame]): A path to a data file (CSV, Excel, JSON)
                or a pandas DataFrame.

        Returns:
            OrchestrationResult: The result of the pipeline run.

        Raises:
            AKDataError: If any pipeline step fails.
        """
        logger.info("Starting AKDATA Orchestration lifecycle...")

        # 1. Load Data
        df = self._load_data(data)
        logger.info(f"Dataset loaded. Initial shape: {df.shape}")

        # 2. Analyze Raw Dataset & Calculate Health Score
        from akdata.analysis.eda import get_eda_summary
        from akdata.analysis.health_score import calculate_health_score

        eda_summary = get_eda_summary(df)
        raw_health_score = calculate_health_score(df)
        logger.info(f"Initial Dataset Health Score: {raw_health_score:.2f}/100")

        # 3. Clean Dataset (Standardize Column Names, Strip Whitespace, Remove Empty Rows)
        from akdata.preprocessing.cleaning import DataCleaner
        cleaner = DataCleaner()
        df = cleaner.fit_transform(df)

        # 4. Duplicate Check/Removal
        if self.config.remove_duplicates:
            from akdata.preprocessing.duplicates import DuplicateHandler
            duplicate_handler = DuplicateHandler()
            df = duplicate_handler.fit_transform(df)

        # 5. Build Preprocessing Pipeline Steps
        # Note: Scaling & Split will be handled as part of the train-test lifecycle
        # to prevent data leakage.
        self._build_pipeline()

        # Extract target if specified
        target = self.config.target_column
        y = None
        X = df.copy()

        if target:
            if target not in df.columns:
                raise ValidationError(f"Target column '{target}' not found in dataset.")
            y = df[target]
            X = df.drop(columns=[target])

        # 6. Fit and Transform Preprocessing Pipeline
        X_train, X_test, y_train, y_test = None, None, None, None
        df_clean = None

        if self.config.split_data and target:
            # 7. Split data before target-dependent transformations
            from akdata.split.train_test import train_test_split_wrapper
            X_train, X_test, y_train, y_test = train_test_split_wrapper(
                X, y, test_size=self.config.test_size,
                stratify=self.config.stratify,
                random_state=self.config.random_state
            )
            
            # Fit & Transform Pipeline on train data
            logger.info("Fitting and transforming Training split...")
            X_train = self.pipeline.fit_transform(X_train, y_train)
            
            # Transform Test data (no fitting to prevent leakage)
            logger.info("Transforming Test split...")
            X_test = self.pipeline.transform(X_test)
            
            # 8. Post-Preprocessing Validation (e.g. Target Leakage detection)
            if self.config.check_leakage:
                from akdata.validation.leakage import detect_leakage
                detect_leakage(X_train, y_train, threshold=self.config.leakage_threshold)

        else:
            logger.info("Processing entire dataset (no train-test split)...")
            df_clean = self.pipeline.fit_transform(X, y)
            if target:
                df_clean[target] = y

        # 9. Calculate final clean health score
        # Reconstruct clean df for evaluation
        df_eval = df_clean
        if df_eval is None and X_train is not None:
            df_eval = X_train.copy()
            if y_train is not None:
                df_eval[target] = y_train

        clean_health_score = calculate_health_score(df_eval) if df_eval is not None else raw_health_score
        logger.info(f"Cleaned Dataset Health Score: {clean_health_score:.2f}/100")

        # 10. Generate Recommendations
        from akdata.analysis.recommendations import generate_recommendations
        recommendations = generate_recommendations(df, self.config, self.pipeline.meta_info)

        # 11. Generate Reports
        report_paths = {}
        if self.config.generate_html_report or self.config.generate_pdf_report:
            report_paths = self._generate_reports(
                df_raw=df,
                df_clean=df_eval if df_eval is not None else df,
                raw_health_score=raw_health_score,
                clean_health_score=clean_health_score,
                eda_summary=eda_summary,
                recommendations=recommendations
            )

        logger.info("AKDATA Orchestration completed successfully.")
        return OrchestrationResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            df_clean=df_clean,
            df_raw=df,
            pipeline=self.pipeline,
            config=self.config,
            raw_health_score=raw_health_score,
            clean_health_score=clean_health_score,
            report_paths=report_paths,
            eda_summary=eda_summary,
            recommendations=recommendations
        )

    def _load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Loads data from a path or passes a DataFrame through.

        Args:
            data (Union[str, pd.DataFrame]): File path or DataFrame.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        if isinstance(data, str):
            from akdata.io.csv import load_csv
            from akdata.io.excel import load_excel
            from akdata.io.json import load_json

            lower_path = data.lower()
            if lower_path.endswith(".csv"):
                return load_csv(data)
            elif lower_path.endswith((".xlsx", ".xls")):
                return load_excel(data)
            elif lower_path.endswith(".json"):
                return load_json(data)
            else:
                raise AKDataError(f"Unsupported file format: {data}")
        
        raise AKDataError(f"Invalid data input type: {type(data)}")

    def _build_pipeline(self) -> None:
        """Builds the preprocessing pipeline steps based on the configuration."""
        self.pipeline = Pipeline()

        # Step A: Missing value Imputation
        if self.config.impute_missing:
            from akdata.preprocessing.missing import MissingValueImputer
            self.pipeline.add_step(
                "imputation",
                MissingValueImputer(
                    numeric_strategy=self.config.numeric_imputation_strategy,
                    categorical_strategy=self.config.categorical_imputation_strategy,
                    fill_value=self.config.impute_constant_value
                )
            )

        # Step B: Outliers handling
        if self.config.handle_outliers:
            from akdata.preprocessing.outliers import OutlierHandler
            self.pipeline.add_step(
                "outliers",
                OutlierHandler(
                    method=self.config.outlier_method,
                    threshold=self.config.outlier_threshold
                )
            )

        # Step C: Datatype detection & casting
        if self.config.auto_cast_dtypes:
            from akdata.preprocessing.datatype import DataTypeCaster
            self.pipeline.add_step("datatype_cast", DataTypeCaster())

        # Step D: Categorical Encoding
        if self.config.encode_categorical:
            from akdata.preprocessing.encoding import CategoricalEncoder
            self.pipeline.add_step(
                "encoding",
                CategoricalEncoder(strategy=self.config.encoding_strategy)
            )

        # Step E: Feature Engineering
        if self.config.feature_engineering:
            from akdata.features.engineering import FeatureEngineer
            self.pipeline.add_step("feature_engineering", FeatureEngineer())

        # Step F: Feature Selection
        if self.config.feature_selection:
            from akdata.features.selection import FeatureSelector
            self.pipeline.add_step(
                "feature_selection",
                FeatureSelector(
                    method=self.config.feature_selection_method,
                    correlation_threshold=self.config.correlation_threshold,
                    variance_threshold=self.config.variance_threshold
                )
            )

        # Step G: Scaling
        if self.config.scale_numeric:
            from akdata.preprocessing.scaling import NumericScaler
            self.pipeline.add_step(
                "scaling",
                NumericScaler(strategy=self.config.scaling_strategy)
            )

    def _generate_reports(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        raw_health_score: float,
        clean_health_score: float,
        eda_summary: Dict[str, Any],
        recommendations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """Generates analysis and quality reports.

        Args:
            df_raw (pd.DataFrame): Raw DataFrame.
            df_clean (pd.DataFrame): Cleaned DataFrame.
            raw_health_score (float): Score before cleaning.
            clean_health_score (float): Score after cleaning.
            eda_summary (Dict[str, Any]): Summary of dataset.
            recommendations (Optional[List[Dict[str, Any]]]): Recommendations list.

        Returns:
            Dict[str, str]: Paths of generated reports.
        """
        paths = {}
        
        # Summary report
        from akdata.reports.summary import generate_summary_report
        summary_txt = generate_summary_report(
            df_raw, df_clean, raw_health_score, clean_health_score, eda_summary
        )
        logger.info(summary_txt)

        # HTML report
        if self.config.generate_html_report:
            from akdata.reports.html import HTMLReporter
            reporter = HTMLReporter(output_dir=self.config.report_dir)
            html_path = reporter.generate(
                df_raw=df_raw,
                df_clean=df_clean,
                raw_health_score=raw_health_score,
                clean_health_score=clean_health_score,
                eda_summary=eda_summary,
                pipeline_meta=self.pipeline.meta_info,
                recommendations=recommendations
            )
            paths["html"] = html_path
            logger.info(f"HTML Report generated at: {html_path}")

        # PDF report
        if self.config.generate_pdf_report:
            from akdata.reports.pdf import PDFReporter
            reporter = PDFReporter(output_dir=self.config.report_dir)
            pdf_path = reporter.generate(
                df_raw=df_raw,
                df_clean=df_clean,
                raw_health_score=raw_health_score,
                clean_health_score=clean_health_score,
                eda_summary=eda_summary,
                pipeline_meta=self.pipeline.meta_info,
                recommendations=recommendations
            )
            if pdf_path:
                paths["pdf"] = pdf_path
                logger.info(f"PDF Report generated at: {pdf_path}")

        return paths
