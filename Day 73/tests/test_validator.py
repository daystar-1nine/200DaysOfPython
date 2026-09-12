"""
Unit tests for dataset validator module.
"""
import pytest
import numpy as np
import pandas as pd
from app.validator import validate_data

def test_validate_valid_dataset(synthetic_sales_df):
    res = validate_data(synthetic_sales_df)
    assert res["valid"] is True
    assert res["n_rows"] == 100
    assert res["feature_variance"] > 0
    assert res["target_variance"] > 0

def test_validate_missing_feature_column():
    df = pd.DataFrame({"Wrong_Col": [1, 2], "Sales": [10, 20]})
    with pytest.raises(KeyKey := KeyError, match="Feature column"):
        validate_data(df)

def test_validate_missing_target_column():
    df = pd.DataFrame({"Advertising_Spend": [1, 2], "Wrong_Col": [10, 20]})
    with pytest.raises(KeyError, match="Target column"):
        validate_data(df)

def test_validate_too_few_rows():
    df = pd.DataFrame({"Advertising_Spend": [1, 2], "Sales": [10, 20]})
    with pytest.raises(ValueError, match="at least 10 rows"):
        validate_data(df, min_rows=10)

def test_validate_zero_variance_feature():
    df = pd.DataFrame({
        "Advertising_Spend": [50.0] * 20,
        "Sales": np.arange(20) * 10
    })
    with pytest.raises(ValueError, match="zero variance"):
        validate_data(df)
