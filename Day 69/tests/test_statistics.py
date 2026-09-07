"""
Tests for stats_calc module.
"""
import pytest
import numpy as np
from app.stats_calc import compute_descriptive_stats

def test_descriptive_stats_exact_mean(exact_mean_data):
    res = compute_descriptive_stats(exact_mean_data)
    assert res["count"] == 5
    assert res["mean"] == 30.0
    assert res["median"] == 30.0
    assert res["min"] == 10.0
    assert res["max"] == 50.0

def test_descriptive_stats_sample_std(exact_mean_data):
    res = compute_descriptive_stats(exact_mean_data)
    # Variance with ddof=1 is 250.0 -> sqrt(250) ~ 15.8114
    expected_std = np.std(exact_mean_data, ddof=1)
    assert np.isclose(res["std_dev"], expected_std, atol=1e-4)

def test_descriptive_stats_standard_error(exact_mean_data):
    res = compute_descriptive_stats(exact_mean_data)
    expected_se = np.std(exact_mean_data, ddof=1) / np.sqrt(5)
    assert np.isclose(res["standard_error"], expected_se, atol=1e-4)

def test_descriptive_stats_iqr(exact_mean_data):
    res = compute_descriptive_stats(exact_mean_data)
    assert res["q3"] >= res["q1"]
    assert res["iqr"] == round(res["q3"] - res["q1"], 4)
