"""Tests for percentiles module."""
import numpy as np
import pandas as pd
from app.statistics.percentiles import calculate_percentile, calculate_percentiles

def test_percentile_p50_equals_median(sample_odd_data):
    p50 = calculate_percentile(sample_odd_data, 50)
    assert p50 == 30.0

def test_percentiles_dict(sample_even_data):
    res = calculate_percentiles(sample_even_data, [25, 50, 75])
    assert "P25" in res
    assert "P50" in res
    assert "P75" in res
    assert res["P25"] < res["P50"] < res["P75"]

def test_percentile_empty(empty_data):
    assert np.isnan(calculate_percentile(empty_data, 90))
