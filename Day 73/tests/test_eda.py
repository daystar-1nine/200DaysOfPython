"""
Unit tests for exploratory data analysis module.
"""
import pytest
import numpy as np
from app.eda import compute_eda

def test_compute_eda_statistics(synthetic_sales_df):
    res = compute_eda(synthetic_sales_df)
    assert res["count"] == 100
    assert res["min_x"] < res["mean_x"] < res["max_x"]
    assert res["min_y"] < res["mean_y"] < res["max_y"]
    assert 0.90 <= res["pearson_r"] <= 1.0
    assert res["p_value"] < 0.001

def test_compute_eda_std_positive(synthetic_sales_df):
    res = compute_eda(synthetic_sales_df)
    assert res["std_x"] > 0
    assert res["std_y"] > 0

def test_compute_eda_perfect_correlation(clean_linear_arrays):
    X, y = clean_linear_arrays
    df = type("df", (), {"values": X})
    import pandas as pd
    df_test = pd.DataFrame({"Advertising_Spend": X, "Sales": y})
    res = compute_eda(df_test)
    assert np.isclose(res["pearson_r"], 1.0)
    assert np.isclose(res["p_value"], 0.0, atol=1e-10)

def test_compute_eda_negative_correlation():
    import pandas as pd
    df = pd.DataFrame({"Advertising_Spend": [1, 2, 3, 4, 5], "Sales": [50, 40, 30, 20, 10]})
    res = compute_eda(df)
    assert np.isclose(res["pearson_r"], -1.0)
