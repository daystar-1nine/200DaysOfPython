"""Tests for dispersion module."""
import numpy as np
import pandas as pd
from app.statistics.dispersion import calculate_range, calculate_variance, calculate_std, calculate_iqr

def test_range_normal(sample_odd_data):
    assert calculate_range(sample_odd_data) == 40.0

def test_range_constant(constant_data):
    assert calculate_range(constant_data) == 0.0

def test_variance_sample_vs_population():
    s = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    var_pop = calculate_variance(s, ddof=0)
    var_sample = calculate_variance(s, ddof=1)
    assert var_pop == 8.0
    assert var_sample == 10.0

def test_std_sample():
    s = pd.Series([10.0, 20.0, 30.0])
    std_val = calculate_std(s, ddof=1)
    assert np.isclose(std_val, 10.0)

def test_iqr_normal():
    s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])
    iqr_val = calculate_iqr(s)
    assert iqr_val > 0
