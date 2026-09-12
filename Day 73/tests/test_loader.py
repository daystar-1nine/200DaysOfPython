"""
Unit tests for data loader module.
"""
import pytest
import pandas as pd
from app.loader import load_csv

def test_load_valid_csv(tmp_path):
    p = tmp_path / "valid.csv"
    p.write_text("Advertising_Spend,Sales\n100,200\n300,400", encoding="utf-8")
    df = load_csv(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

def test_load_non_existent_file():
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file_abc.csv")

def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        load_csv(str(p))

def test_load_single_row_csv(tmp_path):
    p = tmp_path / "single.csv"
    p.write_text("A,B\n1,2", encoding="utf-8")
    df = load_csv(str(p))
    assert len(df) == 1

def test_load_whitespace_csv(tmp_path):
    p = tmp_path / "spaces.csv"
    p.write_text("A , B \n 10 , 20 \n 30 , 40 ", encoding="utf-8")
    df = load_csv(str(p))
    assert len(df) == 2
