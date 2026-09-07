"""
Tests for confidence_intervals module.
"""
import pytest
import numpy as np
from app.confidence_intervals import ConfidenceIntervalEngine

def test_z_interval_mean_accuracy():
    res = ConfidenceIntervalEngine.z_interval_mean(mean=100.0, population_std=15.0, sample_size=100, confidence=0.95)
    # SE = 15 / 10 = 1.5, z = 1.960 -> ME = 2.94
    assert np.isclose(res["standard_error"], 1.5)
    assert np.isclose(res["margin_of_error"], 1.960 * 1.5, atol=0.01)
    assert np.isclose(res["lower_bound"], 100.0 - res["margin_of_error"])
    assert np.isclose(res["upper_bound"], 100.0 + res["margin_of_error"])

def test_t_interval_mean_df(small_sample):
    res = ConfidenceIntervalEngine.t_interval_mean(small_sample, confidence=0.95)
    assert res["degrees_of_freedom"] == len(small_sample) - 1
    assert res["sample_size"] == 5

def test_adaptive_mean_interval_selects_z_when_sigma_known(small_sample):
    res = ConfidenceIntervalEngine.adaptive_mean_interval(small_sample, confidence=0.95, population_std=3.0)
    assert "Z-Interval" in res["method"]

def test_adaptive_mean_interval_selects_t_when_sigma_unknown(small_sample):
    res = ConfidenceIntervalEngine.adaptive_mean_interval(small_sample, confidence=0.95)
    assert "T-Interval" in res["method"]

def test_confidence_level_width_monotonicity(sample_data):
    res_90 = ConfidenceIntervalEngine.t_interval_mean(sample_data, confidence=0.90)
    res_95 = ConfidenceIntervalEngine.t_interval_mean(sample_data, confidence=0.95)
    res_99 = ConfidenceIntervalEngine.t_interval_mean(sample_data, confidence=0.99)
    
    assert res_99["interval_width"] > res_95["interval_width"] > res_90["interval_width"]

def test_proportion_interval_wilson_bounds():
    res = ConfidenceIntervalEngine.proportion_interval(successes=320, total=500, confidence=0.95, method="wilson")
    assert res["sample_proportion"] == 0.64
    assert 0.0 <= res["lower_bound"] <= res["sample_proportion"] <= res["upper_bound"] <= 1.0

def test_proportion_interval_wald_bounds():
    res = ConfidenceIntervalEngine.proportion_interval(successes=320, total=500, confidence=0.95, method="wald")
    assert np.isclose(res["sample_proportion"], 0.64)
    assert res["lower_bound"] < res["sample_proportion"] < res["upper_bound"]
