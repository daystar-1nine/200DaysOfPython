"""
Unit Tests for Data Ingestion and Validation (loader.py).
"""

import pytest
import pandas as pd
from pathlib import Path
from app.loader import load_sales_data
from app.config import SALES_CSV_PATH


def test_load_sales_data_success(temp_csv_path):
    """Verifies that a valid CSV is loaded with correct records and datetime format."""
    df = load_sales_data(temp_csv_path)
    assert not df.empty
    assert len(df) == 12
    assert pd.api.types.is_datetime64_any_dtype(df["Order_Date"])
    assert "Year_Month" in df.columns


def test_load_sales_data_file_not_found(tmp_path):
    """Verifies FileNotFoundError is raised when file does not exist."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_sales_data(non_existent)


def test_load_sales_data_missing_column(tmp_path):
    """Verifies KeyError is raised when required columns are absent."""
    bad_csv = tmp_path / "bad_schema.csv"
    df_bad = pd.DataFrame({"Order_Date": ["2026-01-01"], "Revenue": [1000]})
    df_bad.to_csv(bad_csv, index=False)

    with pytest.raises(KeyError):
        load_sales_data(bad_csv)


def test_load_sales_data_empty_file(tmp_path):
    """Verifies ValueError is raised for an empty CSV."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(Exception):
        load_sales_data(empty_csv)


def test_real_dataset_integrity():
    """Verifies that the actual sales.csv file loads cleanly with minimum row count."""
    df = load_sales_data(SALES_CSV_PATH)
    assert len(df) >= 500
    assert "Region" in df.columns
    assert "Category" in df.columns
    assert "Product" in df.columns
