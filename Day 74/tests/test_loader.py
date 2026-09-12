import pytest
import pandas as pd
import os

try:
    from app.loader import load_data
except ImportError:
    from loader import load_data

def test_load_data_returns_dataframe(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)

def test_load_data_has_required_columns(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    df = load_data(str(p))
    expected_cols = [
        'Record_ID', 'Date', 'TV_Spend', 'Digital_Spend', 'Radio_Spend',
        'Discount', 'Quantity', 'Region', 'Category', 'Competitor_Price',
        'Customer_Count', 'Advertising_Spend', 'Sales', 'Profit', 'Month', 'Year'
    ]
    for col in expected_cols:
        assert col in df.columns

def test_load_data_not_empty(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    df = load_data(str(p))
    assert len(df) > 0

def test_load_data_invalid_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("invalid/path/that/does/not/exist.csv")

def test_load_data_row_count(tmp_path, sample_df):
    # Mocking an 800+ row dataframe for the row count test
    large_df = pd.concat([sample_df]*20, ignore_index=True)
    p = tmp_path / "large_data.csv"
    large_df.to_csv(p, index=False)
    df = load_data(str(p))
    assert len(df) >= 800
