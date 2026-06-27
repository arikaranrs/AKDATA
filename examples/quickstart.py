"""Quickstart example demonstrating end-to-end preprocessing with AKDATA."""

import os
import pandas as pd
import numpy as np
from akdata import ak

# 1. Create a dummy dirty dataset
data = {
    " Name ": [" Alice ", " Bob ", " Charlie ", " Alice ", np.nan],  # Whitespace, duplicate, missing row
    " Age": [25.0, 30.0, 100.0, 30.0, np.nan],                 # Outlier, duplicate, missing row
    " Joining Date ": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-02", np.nan], # Datetime
    " Department": ["HR", "Engineering", "Marketing", "Engineering", np.nan],   # Categorical
    "Salary": [70000, 80000, 75000, 90000, np.nan]            # Target column
}

df_dirty = pd.DataFrame(data)
csv_path = "employee_dirty.csv"
df_dirty.to_csv(csv_path, index=False)
print(f"Generated sample dirty dataset at: {csv_path}")

try:
    # 2. Load the dataset using AKDATA IO
    df = ak.read_csv(csv_path)

    # 3. Process end-to-end using the simple Developer API
    # Target column is 'Salary' (case-insensitive because DataCleaner will lowercase it to 'salary')
    result = ak.prepare(
        df,
        target="salary",
        split_data=True,
        test_size=0.25,
        check_leakage=False,
        generate_html_report=True,
        generate_pdf_report=True
    )

    # 4. Display Results using the stable Public API contract
    print("\nExecuting OrchestrationResult.summary()...")
    result.summary()

    print("\nExporting HTML and PDF reports via Public API methods...")
    html_path = result.export_html()
    pdf_path = result.export_pdf()
    
    print(f"  - HTML report exported to: {os.path.abspath(html_path)}")
    if pdf_path:
        print(f"  - PDF report exported to: {os.path.abspath(pdf_path)}")

finally:
    # Clean up dummy file
    if os.path.exists(csv_path):
        os.remove(csv_path)
