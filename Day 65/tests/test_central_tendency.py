"""Tests for central tendency module."""
import numpy as np
import pandas as pd
from app.statistics.central_tendency import calculate_mean, calculate_median, calculate_mode

def test_mean_normal(sample_odd_data):
    assert calculate_mean(sample_odd_data) == 30.0

def test_mean_empty(empty_data):
    assert np.isnan(calculate_mean(empty_data))

def test_median_odd(sample_odd_data):
    assert calculate_median(sample_odd_data) == 30.0

def test_median_even(sample_even_data):
    assert calculate_median(sample_even_data) == 35.0

def test_mode_single():
    s = pd.Series([1, 2, 2, 3, 4])
    assert calculate_mode(s) == [2]

def test_mode_multiple():
    s = pd.Series([1, 1, 2, 2, 3])
    modes = calculate_mode(s)
    assert sorted(modes) == [1, 2]

def test_mode_empty(empty_data):
    assert calculate_mode(empty_data) == []
