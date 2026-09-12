"""
Unit tests for data cleaner module.
"""
import pytest
import numpy as np
import pandas as pd
from app.cleaner import clean_sales_data

def test_clean_drops_duplicates():
    df = pd.DataFrame({
        "Advertising_Spend": [100, 100, 200],
        "Sales": [200, 200, 400]
    })
    cleaned = clean_sales_data(df)
    assert len(cleaned) == 2

def test_clean_drops_nulls():
    df = pd.DataFrame({
        "Advertising_Spend": [100, np.nan, 300],
        "Sales": [200, 300, np.nan]
    })
    cleaned = clean_sales_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Advertising_Spend"] == 100

def test_clean_filters_negative_values():
    df = pd.DataFrame({
        "Advertising_Spend": [-50, 100, 200],
        "Sales": [100, -200, 400]
    })
    cleaned = clean_sales_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Advertising_Spend"] == 200

def test_clean_filters_zeros():
    df = pd.DataFrame({
        "Advertising_Spend": [0, 100],
        "Sales": [100, 200]
    })
    cleaned = clean_sales_data(df)
    assert len(cleaned) == 1

def test_clean_raises_when_all_invalid():
    df = pd.DataFrame({
        "Advertising_Spend": [-10, -20],
        "Sales": [-50, -60]
    })
    with pytest.raises(ValueError, match="no valid positive records"):
        clean_sales_data(df)
