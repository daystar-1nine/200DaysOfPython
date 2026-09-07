"""
Tests for bootstrap module.
"""
import pytest
import numpy as np
from app.bootstrap import BootstrapEstimator

def test_bootstrap_initialization(sample_data):
    est = BootstrapEstimator(sample_data, seed=42)
    assert len(est.data) == len(sample_data)

def test_bootstrap_estimate_mean(sample_data):
    est = BootstrapEstimator(sample_data, seed=42)
    res = est.estimate(statistic=np.mean, iterations=2_000, confidence=0.95)
    
    assert res["iterations"] == 2_000
    assert np.isclose(res["observed_statistic"], np.mean(sample_data))
    assert res["bootstrap_se"] > 0.0
    assert res["lower_bound"] < res["observed_statistic"] < res["upper_bound"]

def test_bootstrap_estimate_median(sample_data):
    est = BootstrapEstimator(sample_data, seed=42)
    res = est.estimate(statistic=np.median, iterations=2_000, confidence=0.90)
    
    assert np.isclose(res["observed_statistic"], np.median(sample_data))
    assert res["lower_bound"] <= res["observed_statistic"] <= res["upper_bound"]

def test_bootstrap_width_ordering(sample_data):
    est = BootstrapEstimator(sample_data, seed=42)
    res_90 = est.estimate(statistic=np.mean, iterations=2_000, confidence=0.90)
    res_99 = est.estimate(statistic=np.mean, iterations=2_000, confidence=0.99)
    
    assert res_99["interval_width"] > res_90["interval_width"]

def test_bootstrap_reproducibility(sample_data):
    est1 = BootstrapEstimator(sample_data, seed=123)
    est2 = BootstrapEstimator(sample_data, seed=123)
    
    r1 = est1.estimate(statistic=np.mean, iterations=500, confidence=0.95)
    r2 = est2.estimate(statistic=np.mean, iterations=500, confidence=0.95)
    
    np.testing.assert_array_equal(r1["distribution"], r2["distribution"])
