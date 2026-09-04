"""
Module: loader.py
Handles robust CSV data ingestion and schema validation for Student Performance Analyzer V2.
"""

# What is used: Import os module and pathlib Path class.
# Why it is used: Ensures cross-platform file path resolution.
# How it works: Interacts with OS file paths safely.
import os
from pathlib import Path

# What is used: Import pandas library.
# Why it is used: Fundamental dependency for CSV file reading.
# How it works: Provides pd.read_csv function to read tabular files into DataFrames.
import pandas as pd

# Expected schema for raw student CSV file
REQUIRED_COLUMNS = ["Student_ID", "Name", "Department", "Math", "Physics", "Chemistry"]


def load_student_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw student dataset from CSV file and validate schema.

    Args:
        file_path: Path to CSV dataset file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If required columns are missing from the CSV file.
    """
    # What is used: Path object conversion.
    # Why it is used: Normalizes string paths to Path objects for robust existence check.
    # How it works: Constructs Path instance from input parameter.
    path = Path(file_path)

    # What is used: path.exists() validation.
    # Why it is used: Prevents crashing with unhandled file missing errors.
    # How it works: Checks filesystem status for target path.
    if not path.exists():
        raise FileNotFoundError(f"Student CSV file not found at location: {file_path}")

    # What is used: pd.read_csv() with na_values.
    # Why it is used: Loads CSV file into a DataFrame while normalizing common missing value representations.
    # How it works: Parses CSV text into memory table and converts 'NA', 'N/A', empty strings to NaN.
    try:
        df = pd.read_csv(path, na_values=["NA", "N/A", "missing", "-999", ""])
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    # What is used: Set column comparison required_columns - set(df.columns).
    # Why it is used: Ensures input file conforms to expected schema.
    # How it works: Compares required list against DataFrame columns.
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {sorted(list(missing_cols))}")

    return df
