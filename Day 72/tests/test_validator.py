"""
Unit tests for dataset validator module.
"""
import pytest
import pandas as pd
import numpy as np
from app.validator import validate_numeric_data

def test_validate_valid_dataset():
    df = pd.DataFrame({
        "A": np.arange(15),
        "B": np.arange(15) * 2
    })
    res = validate_numeric_data(df, min_rows=10, min_cols=2)
    assert res["valid"] is True
    assert res["n_rows"] == 15
    assert res["n_cols"] == 2
    assert len(res["constant_columns"]) == 0

def test_validate_too_few_rows():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError, match="at least 10 rows"):
        validate_numeric_data(df, min_rows=10)

def test_validate_too_few_cols():
    df = pd.DataFrame({"A": np.arange(15)})
    with pytest.raises(ValueError, match="at least 2 columns"):
        validate_numeric_data(df, min_cols=2)

def test_validate_detects_constant_column():
    df = pd.DataFrame({
        "A": np.arange(15),
        "B": [5.0] * 15
    })
    res = validate_numeric_data(df)
    assert res["valid"] is False
    assert "B" in res["constant_columns"]

def test_validate_multiple_constant_columns():
    df = pd.DataFrame({
        "A": [1.0] * 20,
        "B": [2.0] * 20,
        "C": np.arange(20)
    })
    res = validate_numeric_data(df)
    assert res["valid"] is False
    assert set(res["constant_columns"]) == {"A", "B"}

def test_validate_column_names_returned():
    df = pd.DataFrame({
        "Alpha": np.arange(12),
        "Beta": np.arange(12) * 3
    })
    res = validate_numeric_data(df)
    assert res["columns"] == ["Alpha", "Beta"]

