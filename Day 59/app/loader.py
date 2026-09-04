"""
Module: loader.py
Handles CSV raw dataset ingestion and schema column presence validation.
"""

# What is used: Import os module, pathlib Path, and pandas library.
# Why it is used: Manages file path validation and CSV DataFrame loading.
# How it works: Checks file existence and reads dataset into pandas DataFrame.
import os
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_ID", "Customer_Name",
    "Region", "Category", "Product", "Quantity", "Unit_Price",
    "Discount", "Cost_Price"
]


def load_raw_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Ingest raw transactional CSV sales dataset and validate schema structure.

    Args:
        file_path: Path to raw CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.

    Raises:
        FileNotFoundError: If target file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Target raw dataset file not found at: {path}")

    # What is used: pd.read_csv().
    # Why it is used: Ingests raw sales dataset from disk into memory.
    # How it works: Serializes CSV file rows into pandas DataFrame object.
    df = pd.read_csv(path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required schema columns: {missing_cols}")

    return df
