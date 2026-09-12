import pytest
import pandas as pd
from unittest.mock import patch

class DataLoader:
    def __init__(self, config):
        self.config = config
    def load_data(self):
        return pd.read_csv(self.config.data_path)

def test_load_data_returns_dataframe(config):
    loader = DataLoader(config)
    df_mock = pd.DataFrame({'Sales': [1, 2, 3]})
    with patch('pandas.read_csv', return_value=df_mock):
        df = loader.load_data()
        assert isinstance(df, pd.DataFrame)

def test_load_data_required_columns(config):
    loader = DataLoader(config)
    df_mock = pd.DataFrame({'Sales': [1, 2, 3], 'Date': ['2021-01-01', '2021-01-02', '2021-01-03']})
    with patch('pandas.read_csv', return_value=df_mock):
        df = loader.load_data()
        assert 'Sales' in df.columns

def test_load_data_not_empty(config):
    loader = DataLoader(config)
    df_mock = pd.DataFrame({'Sales': [1, 2, 3]})
    with patch('pandas.read_csv', return_value=df_mock):
        df = loader.load_data()
        assert not df.empty

def test_load_data_file_not_found(config):
    loader = DataLoader(config)
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            loader.load_data()

def test_load_data_row_count(config):
    loader = DataLoader(config)
    df_mock = pd.DataFrame({'Sales': [1]*100})
    with patch('pandas.read_csv', return_value=df_mock):
        df = loader.load_data()
        assert len(df) == 100
