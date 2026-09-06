"""Tests for outlier detection module."""
import pandas as pd
from app.statistics.outliers import detect_iqr_outliers

def test_outliers_none():
    s = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    res = detect_iqr_outliers(s)
    assert res["count"] == 0
    assert res["percentage"] == 0.0

def test_outliers_presence(sample_outlier_data):
    res = detect_iqr_outliers(sample_outlier_data)
    assert res["count"] == 2
    assert 200.0 in res["outliers"]
    assert -50.0 in res["outliers"]

def test_outliers_custom_multiplier():
    s = pd.Series([10, 12, 14, 16, 18, 35])
    res_1_5 = detect_iqr_outliers(s, multiplier=1.5)
    res_3_0 = detect_iqr_outliers(s, multiplier=3.0)
    assert res_1_5["count"] >= res_3_0["count"]
