"""Tests for zscore module."""
import numpy as np
import pandas as pd
from app.statistics.zscore import calculate_zscores, detect_zscore_outliers

def test_zscore_mean_is_zero():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    z = calculate_zscores(s)
    assert np.isclose(z.mean(), 0.0)

def test_zscore_constant_safe(constant_data):
    z = calculate_zscores(constant_data)
    assert (z == 0.0).all()

def test_zscore_outlier_screening():
    s = pd.Series([10, 11, 10, 12, 11, 10, 11, 100])
    res = detect_zscore_outliers(s, threshold=2.0)
    assert res["count"] == 1
    assert 100 in res["outliers"]
