import pytest
import pandas as pd
from app.data.loader import load_raw_data

def test_load_raw_data_success(sample_config):
    df = load_raw_data(sample_config.raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 100

def test_load_raw_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw_data('non_existent_path.csv')

def test_load_raw_data_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("")
    with pytest.raises(ValueError):
        load_raw_data(str(f))

def test_load_raw_data_schema(sample_config):
    df = load_raw_data(sample_config.raw_data_path)
    assert 'Churn' in df.columns
    assert 'Customer_ID' in df.columns

def test_load_raw_data_row_count(sample_config):
    df = load_raw_data(sample_config.raw_data_path)
    assert len(df) >= 1000
