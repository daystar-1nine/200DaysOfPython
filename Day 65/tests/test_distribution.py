"""Tests for distribution shape module."""
import numpy as np
import pandas as pd
from app.statistics.distribution import calculate_skewness, calculate_kurtosis, classify_distribution

def test_skewness_symmetric():
    s = pd.Series([10, 20, 30, 40, 50])
    assert np.isclose(calculate_skewness(s), 0.0)

def test_skewness_right_skewed():
    s = pd.Series([10, 12, 14, 15, 16, 18, 20, 100])
    assert calculate_skewness(s) > 1.0

def test_kurtosis_normal():
    s = pd.Series([10, 20, 30, 40, 50])
    kurt = calculate_kurtosis(s)
    assert isinstance(kurt, float)

def test_classify_distribution():
    res = classify_distribution(skewness=1.8, kurtosis=4.5)
    assert "Positively Skewed" in res["skewness_classification"]
    assert "Leptokurtic" in res["kurtosis_classification"]
