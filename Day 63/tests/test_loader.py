"""
Tests for Dataset Loader Module
===============================
"""

import pytest
import pandas as pd
from app.loader import load_dataset

def test_load_dataset_success():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 700
    assert "Revenue" in df.columns
    assert "Profit" in df.columns

def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file_path.csv")

def test_load_dataset_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError) as excinfo:
        load_dataset(str(bad_csv))
    assert "missing required columns" in str(excinfo.value)
