"""IO package for AKDATA.

Contains data loading functions for CSV, Excel, and JSON.
"""

from akdata.io.csv import load_csv
from akdata.io.excel import load_excel
from akdata.io.json import load_json

__all__ = ["load_csv", "load_excel", "load_json"]
