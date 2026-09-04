"""
Unit tests for data loader module app/loader.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 58 parent directory to sys.path.
import sys
from pathlib import Path

DAY58_DIR = Path(__file__).resolve().parent.parent
if str(DAY58_DIR) not in sys.path:
    sys.path.insert(0, str(DAY58_DIR))

# What is used: Import pytest, pandas, and load_data function.
# Why it is used: Tests CSV file ingestion, missing file exception, and schema validation.
# How it works: Runs load_data assertion tests.
import pandas as pd
import pytest
from app.loader import load_data


def test_load_data_valid(sample_csv_file):
    """
    Test loading a valid customer CSV file.
    """
    # What is used: load_data call on valid temporary CSV file.
    # Why it is used: Verifies file reading succeeds and returns DataFrame with mandatory columns.
    # How it works: Asserts DataFrame type and row count.
    df = load_data(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "Customer_ID" in df.columns
    assert "Name" in df.columns


def test_load_data_missing_file_raises_error(tmp_path):
    """
    Test loading a non-existent file path raises FileNotFoundError.
    """
    # What is used: pytest.raises(FileNotFoundError).
    # Why it is used: Verifies proper exception handling when target CSV file is missing.
    # How it works: Catches FileNotFoundError thrown by load_data.
    missing_path = tmp_path / "non_existent_customers.csv"
    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


def test_load_data_missing_columns_raises_error(tmp_path):
    """
    Test loading a CSV file missing mandatory schema columns raises ValueError.
    """
    # What is used: Generating CSV file missing required columns.
    # Why it is used: Asserts schema validation enforcement in loader module.
    # How it works: Writes CSV missing 'Salary' and 'Department' columns.
    invalid_csv = tmp_path / "invalid_schema.csv"
    invalid_df = pd.DataFrame({"Customer_ID": ["C1"], "Name": ["Test"]})
    invalid_df.to_csv(invalid_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_data(invalid_csv)
