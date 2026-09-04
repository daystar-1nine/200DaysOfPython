"""
Module: loader.py
Handles CSV file ingestion and schema validation for Sales Analytics Engine.
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

# Required column schema for raw sales CSV file
REQUIRED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_ID", "Customer_Name",
    "Region", "Category", "Product", "Quantity", "Unit_Price", "Discount"
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw sales CSV dataset from file and validate column schema.

    Args:
        file_path: Path to raw sales CSV file.

    Returns:
        pd.DataFrame: Loaded sales DataFrame.

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
        raise FileNotFoundError(f"Sales dataset CSV not found at location: {file_path}")

    # What is used: pd.read_csv() with na_values.
    # Why it is used: Parses CSV into a DataFrame, mapping missing string tokens to NaN floats.
    # How it works: Reads tabular data and returns pandas DataFrame.
    try:
        df = pd.read_csv(path, na_values=["NA", "N/A", "missing", "-999", ""])
    except Exception as exc:
        raise ValueError(f"Failed to parse sales CSV file: {exc}") from exc

    # What is used: Set column comparison.
    # Why it is used: Validates presence of mandatory schema columns.
    # How it works: Checks if required schema columns exist in df.columns.
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {sorted(list(missing))}")

    return df
