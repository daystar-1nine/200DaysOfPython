import pytest
import os
import pandas as pd
from app.loader import load_raw_data

def test_load_raw_data_valid(sample_config):
    df = load_raw_data(sample_config.raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 100

def test_load_raw_data_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_raw_data('non_existent_file.csv')

def test_load_raw_data_empty(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(Exception):
        load_raw_data(str(empty_file))

def test_load_raw_data_columns(sample_config):
    df = load_raw_data(sample_config.raw_data_path)
    assert 'Churn' in df.columns
    assert 'Customer_ID' in df.columns

def test_load_raw_data_dtypes(sample_config):
    df = load_raw_data(sample_config.raw_data_path)
    assert df.shape[1] >= 10
