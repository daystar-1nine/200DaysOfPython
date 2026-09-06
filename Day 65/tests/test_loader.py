"""Tests for loader module."""
import os
import pytest
import pandas as pd
from app.loader import load_dataset

def test_load_dataset_success(real_sales_df):
    assert isinstance(real_sales_df, pd.DataFrame)
    assert len(real_sales_df) == 750
    assert "Revenue" in real_sales_df.columns

def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file_path.csv")
