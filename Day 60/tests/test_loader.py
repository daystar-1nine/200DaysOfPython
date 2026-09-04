"""
Unit Tests for app/loader.py module.
"""

import pandas as pd
import pytest
from app.loader import load_dataset


def test_load_dataset_success(sample_csv_path):
    df = load_dataset(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert "Order_ID" in df.columns
    assert "Unit_Price" in df.columns


def test_load_dataset_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file_path.csv")


def test_load_dataset_missing_columns(tmp_path):
    invalid_file = tmp_path / "bad.csv"
    pd.DataFrame({"Order_ID": ["O1"], "Bad_Col": [1]}).to_csv(invalid_file, index=False)
    with pytest.raises(ValueError) as exc:
        load_dataset(invalid_file)
    assert "missing required schema columns" in str(exc.value)
