"""
Module: loader.py
Handles CSV dataset loading and schema validation for Day 60 BI Analytics Engine.
"""

# What is used: Import pathlib Path and pandas library.
# Why it is used: Ingests raw sales CSV files and verifies required columns.
# How it works: Checks file existence and validates schema presence against REQUIRED_COLUMNS.
from pathlib import Path
import pandas as pd
from app.config import REQUIRED_COLUMNS


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw CSV dataset into a pandas DataFrame and validate column schema.

    Args:
        file_path: Path to target CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.

    Raises:
        FileNotFoundError: If the target file path does not exist.
        ValueError: If any required schema column is missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Target raw dataset file not found at: {path}")

    # What is used: pd.read_csv().
    # Why it is used: Loads tabular CSV data into memory.
    # How it works: Parses CSV stream into a pandas DataFrame.
    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required schema columns: {missing}")

    return df
