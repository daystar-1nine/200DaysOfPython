"""
Unit Tests for app/loader.py module.
"""

import pandas as pd
import pytest
from app.loader import load_raw_dataset


def test_load_raw_dataset_success(sample_csv_file):
    df = load_raw_dataset(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert "Order_ID" in df.columns
    assert "Unit_Price" in df.columns


def test_load_raw_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_raw_dataset("non_existent_path.csv")


def test_load_raw_dataset_missing_schema_columns(tmp_path):
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"Order_ID": [1, 2], "Wrong_Col": [10, 20]}).to_csv(invalid_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        load_raw_dataset(invalid_csv)
    assert "missing required schema columns" in str(exc_info.value)
