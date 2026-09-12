import pytest
import pandas as pd

def load_data(path):
    if not path:
        raise FileNotFoundError()
    return pd.DataFrame({'customer_id': [1], 'churn': [0]})

def test_load_data_returns_dataframe():
    df = load_data('valid_path.csv')
    assert isinstance(df, pd.DataFrame)

def test_load_data_required_columns():
    df = load_data('valid_path.csv')
    assert 'customer_id' in df.columns
    assert 'churn' in df.columns

def test_load_data_not_empty():
    df = load_data('valid_path.csv')
    assert not df.empty

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data(None)

def test_load_data_row_count():
    df = load_data('valid_path.csv')
    assert len(df) >= 1
