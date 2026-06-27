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

## Why AKDATA?

Machine Learning projects spend a significant portion of development time preparing and cleaning data before model training.

AKDATA automates this process while maintaining transparency, reproducibility, and machine learning best practices.

Instead of manually writing preprocessing code for every dataset, developers can prepare datasets using a single, consistent workflow.

---

## AKDATA Workflow

```text
Raw Dataset
      │
      ▼
Data Loading
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Dataset Profiling
      │
      ▼
Dataset Health Score
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Duplicate Removal
      │
      ▼
Outlier Detection
      │
      ▼
Data Type Conversion
      │
      ▼
Categorical Encoding
      │
      ▼
Feature Engineering
      │
      ▼
Feature Selection
      │
      ▼
Train/Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Data Leakage Validation
      │
      ▼
Professional Report Generation
      │
      ▼
Ready for Machine Learning
```

---

## Example Output

```python
from akdata import ak

dataset = ak.read_csv("employees.csv")

result = ak.prepare(
    dataset,
    target="Salary"
)

print(result.health_score)

print(result.recommendations)

result.summary()

result.export_html()

result.export_pdf()

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

## What AKDATA Automatically Does

✔ Detects missing values

✔ Removes duplicates

✔ Detects outliers

✔ Converts incorrect data types

✔ Encodes categorical columns

✔ Generates new features

✔ Selects useful features

✔ Splits training and testing data

✔ Scales numerical features

✔ Prevents data leakage

✔ Computes Dataset Health Score

✔ Generates intelligent preprocessing recommendations

✔ Produces professional HTML/PDF reports

---

## Roadmap

### Version 1.0

* Core preprocessing engine
* Dataset Health Score
* Recommendation Engine
* HTML Reports
* PDF Reports

### Version 1.1

* Time Series preprocessing
* Feature importance reports

### Version 2.0

* Image dataset preprocessing
* NLP dataset preprocessing
* AutoML integration
* Plugin architecture
* CLI support

```
```

---



## Development

AKDATA follows modern Python development practices and is designed to be maintainable, modular, and production-ready.

Clone the repository:

```bash
git clone https://github.com/arikaranrs/AKDATA.git
cd AKDATA
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the project in editable mode:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Tests

Execute the complete unit test suite:

```bash
pytest
```

To view detailed test output:

```bash
pytest -v
```

---

## Contributing

Contributions are welcome.

If you would like to improve AKDATA:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Add or update tests where appropriate.
5. Submit a Pull Request.

Please read **CONTRIBUTING.md** before submitting contributions.

---

## Roadmap

### Current Version (v1.0)

* Intelligent Data Preprocessing
* Dataset Health Score
* Intelligent Recommendation Engine
* Data Leakage Detection
* HTML Reports
* PDF Reports
* Modular Pipeline Architecture

### Future Releases

* Time Series Support
* NLP Dataset Support
* Image Dataset Support
* AutoML Integration
* Command Line Interface (CLI)
* Plugin System
* REST API
* Interactive Dashboard

---

## License

This project is licensed under the **MIT License**.

See the **LICENSE** file for complete license information.

---

## Author

**Arikaran R**

Creator and Maintainer of **AKDATA**

* GitHub: https://github.com/arikaranrs
* Email: [arikaranr2410@gmail.com](mailto:arikaranr2410@gmail.com)

If you find AKDATA useful, consider giving the repository a ⭐ on GitHub and contributing to its future development.
