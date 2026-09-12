"""
Unit tests for data loader module.
"""
import os
import pytest
import pandas as pd
from app.loader import load_dataset

def test_load_valid_csv(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("A,B\n1,2\n3,4", encoding="utf-8")
    df = load_dataset(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

def test_load_valid_json(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('[{"A": 1, "B": 2}, {"A": 3, "B": 4}]', encoding="utf-8")
    df = load_dataset(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

def test_load_non_existent_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_path_xyz.csv")

def test_load_unsupported_format(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("sample text", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(str(p))

def test_load_empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        load_dataset(str(p))
