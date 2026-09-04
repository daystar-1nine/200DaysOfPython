"""
Module: loader.py
Handles CSV file ingestion and schema validation for Messy Customer Dataset Cleaning Pipeline.
"""

# What is used: Import os module and pathlib Path class.
# Why it is used: Manages cross-platform file paths and existence checks.
# How it works: Inspects target filesystem locations.
import os
from pathlib import Path

# What is used: Import pandas library.
# Why it is used: Fundamental library for reading CSV datasets into DataFrames.
# How it works: Provides pd.read_csv to parse CSV files.
import pandas as pd

# Expected column schema for raw messy customer CSV file
REQUIRED_COLUMNS = [
    "Customer_ID", "Name", "Age", "Gender", "Email",
    "Phone", "City", "Salary", "Join_Date", "Department"
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw messy customer CSV dataset from file and validate column schema.

    Args:
        file_path: Path to raw messy customer CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ValueError: If required columns are missing from the schema.
    """
    # What is used: Path object conversion.
    # Why it is used: Normalizes input path string to Path instance.
    # How it works: Creates Path object from file_path argument.
    path = Path(file_path)

    # What is used: path.exists() check.
    # Why it is used: Guarantees file presence before attempting file read.
    # How it works: Returns True if file exists on disk.
    if not path.exists():
        raise FileNotFoundError(f"Customer CSV dataset not found at location: {file_path}")

    # What is used: pd.read_csv() with na_values.
    # Why it is used: Parses CSV into a DataFrame, mapping missing string tokens to NaN.
    # How it works: Reads tabular data and returns pandas DataFrame.
    try:
        df = pd.read_csv(path, na_values=["NA", "N/A", "missing", "-999", ""])
    except Exception as exc:
        raise ValueError(f"Failed to parse customer CSV file: {exc}") from exc

    # What is used: Set column comparison.
    # Why it is used: Validates presence of mandatory schema columns.
    # How it works: Checks if required schema columns exist in df.columns.
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {sorted(list(missing))}")

    return df
