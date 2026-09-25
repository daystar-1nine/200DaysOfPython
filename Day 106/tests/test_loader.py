"""
Tests for DataLoader in Day 106: RNNs & Sequential Text Learning.
"""

from pathlib import Path
import pytest
import pandas as pd
from app.data.loader import DataLoader
from app.data.validator import DataValidator
from app.config import RAW_DATA_PATH


def test_loader_loads_existing_dataset():
    df = DataLoader.load_csv(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 800


def test_loader_has_required_columns():
    df = DataLoader.load_csv(RAW_DATA_PATH)
    assert "label" in df.columns
    assert "text" in df.columns
    assert "target" in df.columns


def test_loader_valid_labels():
    df = DataLoader.load_csv(RAW_DATA_PATH)
    unique_labels = set(df["label"].unique())
    assert unique_labels == {"ham", "spam"}
    assert set(df["target"].unique()) == {0, 1}


def test_loader_no_nulls():
    df = DataLoader.load_csv(RAW_DATA_PATH)
    assert df["text"].isnull().sum() == 0
    assert df["label"].isnull().sum() == 0


def test_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DataLoader.load_csv("non_existent_file_path.csv")
