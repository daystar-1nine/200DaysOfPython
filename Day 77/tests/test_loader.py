import pytest
import os
import pandas as pd
from app.loader import load_data

def test_load_data_valid(config):
    df = load_data(config.DATA_PATH_RAW, config)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2000

def test_load_data_required_columns(config):
    df = load_data(config.DATA_PATH_RAW, config)
    for col in ["Customer_ID", "Age", "Monthly_Charges", "Churn", "Risk_Level"]:
        assert col in df.columns

def test_load_data_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "non_existent.csv"))

def test_load_data_empty_file(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(Exception):
        load_data(str(empty_csv))
