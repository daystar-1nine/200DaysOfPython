"""
Unit tests for data cleaner module.
"""
import pytest
import numpy as np
import pandas as pd
from app.cleaner import clean_dataset

def test_clean_filters_non_numeric():
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "Salary": [50000.0, 60000.0, 70000.0]
    })
    cleaned = clean_dataset(df)
    assert "Name" not in cleaned.columns
    assert list(cleaned.columns) == ["Age", "Salary"]

def test_clean_drops_nans():
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0],
        "B": [5.0, 6.0, 7.0, 8.0]
    })
    cleaned = clean_dataset(df)
    assert len(cleaned) == 3
    assert not cleaned.isna().any().any()

def test_clean_handles_infinities():
    df = pd.DataFrame({
        "A": [1.0, np.inf, 3.0],
        "B": [4.0, 5.0, -np.inf]
    })
    # Since row 1 has inf and row 2 has -inf, only row 0 survives
    cleaned = clean_dataset(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["A"] == 1.0

def test_clean_raises_if_no_numeric():
    df = pd.DataFrame({"Name": ["A", "B"], "City": ["X", "Y"]})
    with pytest.raises(ValueError, match="No numerical columns found"):
        clean_dataset(df)
