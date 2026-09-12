"""
Unit tests for outlier detection and leverage analysis.
"""
import pytest
import numpy as np
import pandas as pd
from app.outliers import detect_univariate_outliers, evaluate_outlier_impact

def test_detect_univariate_outliers_finds_extreme_point():
    s = pd.Series([10.0, 11.0, 10.5, 9.8, 10.2, 100.0])
    mask = detect_univariate_outliers(s, k=1.5)
    assert mask.iloc[-1] == True
    assert mask.iloc[:-1].sum() == 0

def test_detect_univariate_outliers_none_on_uniform():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mask = detect_univariate_outliers(s, k=1.5)
    assert mask.sum() == 0

def test_evaluate_outlier_impact_flags_high_leverage():
    x = np.linspace(1, 30, 30)
    y = 2.0 * x + 5.0
    x_c = np.append(x, 200.0)
    y_c = np.append(y, -200.0)
    df = pd.DataFrame({"X": x_c, "Y": y_c})
    
    impact = evaluate_outlier_impact(df, "X", "Y")
    assert impact["n_outliers"] >= 1
    assert impact["has_high_leverage"] is True
    assert impact["delta_r"] >= 0.10

def test_evaluate_outlier_impact_clean_data_low_delta():
    x = np.linspace(1, 40, 40)
    y = 3.0 * x + 2.0
    df = pd.DataFrame({"X": x, "Y": y})
    impact = evaluate_outlier_impact(df, "X", "Y")
    assert impact["has_high_leverage"] is False
    assert impact["delta_r"] < 0.05
