# AKDATA

[![Tests](https://github.com/arikaranrs/AKDATA/actions/workflows/python-package.yml/badge.svg)](https://github.com/arikaranrs/AKDATA/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/akdata/)

**AKDATA** is an enterprise-grade, production-quality, open-source Python library for intelligent and structured data preprocessing, quality profiling, leakage protection, and automated reporting.

Designed to prevent common machine learning bugs, prevent data leakage, and automate tedious data preparation, AKDATA exposes both a simple one-line API and modular pipelines suitable for mission-critical ML workflows.

---

## Key Features

- 🛠 **Automated Data Cleaning**: Strip spaces, standardized snake_case column names, drop completely empty columns and rows.
- 🩺 **Dataset Health Grading**: Compute a robust Data Health Score (0-100) based on missingness, duplicates, outliers, and type conflicts.
- 🔒 **Data Leakage Protection**: Automatically detect and flag target leakage (extremely highly predictive variables) prior to model training.
- ⚙ **Fit-Transform Consistency**: Standardized transformers that compute statistics on the training split and apply them cleanly to test splits without leakage.
- 📈 **Automatic Feature Engineering**: Extract components of datetime columns and create interaction metrics safely.
- 📊 **Automated Professional Dashboards**: Export beautiful, self-contained HTML dashboards and PDF summaries.

---

## Installation

```bash
pip install akdata
```

To include PDF export features:
```bash
pip install akdata[pdf]
```

---

## Quickstart

Prepare your raw dataframe for model training in one simple step:

```python
import pandas as pd
from akdata import ak

# Load data safely
df = ak.read_csv("dataset.csv")

# Prepare, clean, profile, split, and validate end-to-end
result = ak.prepare(
    df,
    target="Salary",
    split_data=True,
    test_size=0.2
)

# Train your scikit-learn or gradient-boosted models safely
model.fit(result.X_train, result.y_train)
```

---

## Modular Usage

You can also construct and run custom pipelines manually:

```python
from akdata.core.pipeline import Pipeline
from akdata.preprocessing.missing import MissingValueImputer
from akdata.preprocessing.scaling import NumericScaler

pipeline = Pipeline()
pipeline.add_step("impute", MissingValueImputer(numeric_strategy="median"))
pipeline.add_step("scale", NumericScaler(strategy="standard"))

# Fit on training features
pipeline.fit(X_train)

# Transform training and testing splits independently
X_train_clean = pipeline.transform(X_train)
X_test_clean = pipeline.transform(X_test)
```

---

## Development & Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to get started on submitting pull requests.

### Testing
Run the comprehensive unit test suite:
```bash
pytest
```

---

## License

AKDATA is released under the [MIT License](LICENSE).
