"""Recommendation Engine for AKDATA.

Analyzes dataset properties and outputs explainable preprocessing decisions,
reasons, and alternatives.
"""

import pandas as pd
from typing import Any, Dict, List
from akdata.core.config import PipelineConfig


def generate_recommendations(
    df_raw: pd.DataFrame,
    config: PipelineConfig,
    pipeline_meta: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate detailed, explainable preprocessing recommendations and alternative strategies.

    Args:
        df_raw (pd.DataFrame): The original raw dataset.
        config (PipelineConfig): Configuration settings applied.
        pipeline_meta (Dict[str, Any]): Transformations meta log from pipeline run.

    Returns:
        List[Dict[str, str]]: A list of recommendations containing decision, reason, alternatives, and severity.
    """
    recommendations = []

    # 1. Missing Values Imputation Recommendation
    total_nulls = df_raw.isnull().sum().sum()
    if total_nulls > 0:
        null_percentage = (total_nulls / (df_raw.shape[0] * df_raw.shape[1])) * 100
        
        if config.impute_missing:
            recommendations.append({
                "feature": "Missing Values",
                "decision": f"Imputed missing values using numeric strategy '{config.numeric_imputation_strategy}' and categorical strategy '{config.categorical_imputation_strategy}'.",
                "reason": f"Dataset contains {total_nulls} missing values ({null_percentage:.2f}% of total cells). Imputation was enabled to keep rows for downstream models.",
                "alternatives": "Drop rows containing missing values (use config.impute_missing=False or dropna manually), or use advanced estimators like KNNImputer / IterativeImputer.",
                "severity": "info"
            })
    else:
        recommendations.append({
            "feature": "Missing Values",
            "decision": "No missing values found; skipped imputation.",
            "reason": "The dataset is fully populated.",
            "alternatives": "None needed.",
            "severity": "info"
        })

    # 2. Outlier Treatment Recommendation
    numeric_cols = df_raw.select_dtypes(include=["number"]).columns
    outlier_count = 0
    if len(df_raw) > 0 and len(numeric_cols) > 0:
        for col in numeric_cols:
            series = df_raw[col].dropna()
            if len(series) > 0:
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                outlier_count += int(((series < lower) | (series > upper)).sum())

    if outlier_count > 0:
        if config.handle_outliers:
            recommendations.append({
                "feature": "Outliers",
                "decision": f"Clipped outliers using '{config.outlier_method}' method with factor {config.outlier_threshold}.",
                "reason": f"Detected {outlier_count} values outside normal mathematical distributions. Clipping prevents extreme values from skewing linear algorithms.",
                "alternatives": "Drop rows containing outliers (requires changing outlier action to 'drop'), or keep outliers unmodified if they represent valid rare occurrences.",
                "severity": "warning"
            })
    else:
        recommendations.append({
            "feature": "Outliers",
            "decision": "No extreme outliers detected; skipped outlier treatment.",
            "reason": "All numerical data points reside within normal distribution limits.",
            "alternatives": "None needed.",
            "severity": "info"
        })

    # 3. Categorical Encodings Recommendation
    cat_cols = df_raw.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        if config.encode_categorical:
            recommendations.append({
                "feature": "Categorical Columns",
                "decision": f"Encoded categorical features using strategy '{config.encoding_strategy}'.",
                "reason": f"Machine Learning models require numerical representations. Found {len(cat_cols)} non-numeric features: {cat_cols}.",
                "alternatives": "Use 'label' encoding to minimize dimensionality expansion, target encoding for high-cardinality features, or leave unencoded if using tree models like CatBoost/LightGBM.",
                "severity": "info"
            })

    # 4. Feature Selection Recommendation
    if config.feature_selection:
        dropped_feats = pipeline_meta.get("feature_selection", {}).get("dropped_features", [])
        if dropped_feats:
            recommendations.append({
                "feature": "Feature Selection",
                "decision": f"Dropped {len(dropped_feats)} features: {dropped_feats}.",
                "reason": f"Dropped due to low variance (< {config.variance_threshold}) or high correlation (> {config.correlation_threshold}) to prevent collinearity.",
                "alternatives": "Relax the correlation threshold (e.g. increase correlation_threshold to 0.95), or use feature importance methods (e.g., Random Forest selection).",
                "severity": "suggestion"
            })

    # 5. Data Split Recommendation
    if config.split_data:
        recommendations.append({
            "feature": "Data Splitting",
            "decision": f"Split dataset into Train ({(1.0 - config.test_size)*100:.0f}%) and Test ({config.test_size*100:.0f}%) splits.",
            "reason": "Standard Machine Learning best practice requires an independent test split to measure generalization performance and detect overfitting.",
            "alternatives": "Disable splitting (config.split_data=False) to preprocess the entire dataset as one continuous block, or use K-Fold cross validation.",
            "severity": "info"
        })

    return recommendations
