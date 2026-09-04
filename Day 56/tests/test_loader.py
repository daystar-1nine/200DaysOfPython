"""
Unit tests for data loader module app/loader.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Ensures app package imports resolve cleanly during pytest execution.
# How it works: Appends Day 56 parent path to sys.path.
import sys
from pathlib import Path

DAY56_DIR = Path(__file__).resolve().parent.parent
if str(DAY56_DIR) not in sys.path:
    sys.path.insert(0, str(DAY56_DIR))

# What is used: Import pytest and loader module functions.
# Why it is used: Asserts file reading behaviors and exception handling.
# How it works: Executes load_student_csv tests.
import pandas as pd
import pytest
from app.loader import load_student_csv


def test_load_valid_csv(sample_csv_file):
    """
    Test loading a valid student CSV file.
    """
    # What is used: Function call load_student_csv.
    # Why it is used: Verifies that CSV reads into a valid DataFrame with expected columns.
    # How it works: Compares resulting DataFrame length and column set.
    df = load_student_csv(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "Student_ID" in df.columns
    assert "Math" in df.columns


def test_load_missing_file_raises_error(tmp_path):
    """
    Test loading a non-existent CSV file raises FileNotFoundError.
    """
    # What is used: pytest.raises(FileNotFoundError).
    # Why it is used: Ensures proper exception handling when file does not exist.
    # How it works: Catches FileNotFoundError raised by load_student_csv.
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_student_csv(non_existent)


def test_load_csv_missing_required_columns_raises_error(tmp_path):
    """
    Test loading a CSV file missing required columns raises ValueError.
    """
    # What is used: Generating invalid CSV file with missing columns.
    # Why it is used: Tests schema validation in loader.
    # How it works: Writes CSV missing 'Physics' and 'Chemistry' columns.
    invalid_csv = tmp_path / "invalid.csv"
    invalid_df = pd.DataFrame({"Student_ID": ["S1"], "Name": ["Test"]})
    invalid_df.to_csv(invalid_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_student_csv(invalid_csv)
