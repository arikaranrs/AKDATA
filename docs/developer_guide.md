# AKDATA Developer Guide

Welcome to the internal architectural guide for AKDATA.

## Folder Architecture

- `core/`: Config (`config.py`), Exception definitions (`exceptions.py`), Sequential fit-transform pipelines (`pipeline.py`), and high-level coordinator (`orchestrator.py`).
- `io/`: Delimiter sniffing CSV parser, Excel sheet reader, and JSON converters.
- `analysis/`: Computes Exploratory statistics, Skewness/Kurtosis, and Weighted Dataset Health Score.
- `preprocessing/`: Modular transformers implementing base transformers for duplication dropping, outlier clipping, type casting, imputation, scaling, etc.
- `features/`: Automatic datetime parts engineering and variance/correlation filters.
- `split/`: Reproducible train-test split wrapper with target class balance validations.
- `validation/`: Schema checkers and target leakage correlations.
- `reports/`: Text logging, interactive SVG/HTML dashboards, and PDF build wrappers.

## Adding Custom Steps

To extend the pipeline, inherit from `BaseTransformer` inside `akdata/core/pipeline.py` and implement `fit` and `transform` overrides:

```python
from akdata.core.pipeline import BaseTransformer

class CustomThresholdFilter(BaseTransformer):
    def __init__(self, limit=10.0):
        super().__init__()
        self.limit = limit

    def fit(self, df, y=None):
        self.is_fitted = True
        return self

    def transform(self, df):
        # Apply transformation logic here
        return df.clip(upper=self.limit)
```
