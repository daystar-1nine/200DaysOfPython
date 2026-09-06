"""
Tests for BootstrapEstimator.
"""
import pytest
import numpy as np
from app.bootstrap import BootstrapEstimator

def test_bootstrap_empty_sample_error():
    with pytest.raises(ValueError, match="Sample cannot be empty"):
        BootstrapEstimator(np.array([]))

def test_bootstrap_mean_estimation():
    sample = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    boot = BootstrapEstimator(sample, seed=42)
    res = boot.estimate(statistic_func=np.mean, n_iterations=2_000, ci_level=0.95)
    
    assert res["observed"] == 30.0
    assert np.isclose(res["boot_mean"], 30.0, atol=1.0)
    assert res["boot_se"] > 0.0
    assert res["ci_lower"] < res["observed"] < res["ci_upper"]

def test_bootstrap_median_estimation():
    sample = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 100.0])
    boot = BootstrapEstimator(sample, seed=42)
    res = boot.estimate(statistic_func=np.median, n_iterations=2_000, ci_level=0.90)
    
    assert res["observed"] == 20.0
    assert res["ci_lower"] <= res["observed"] <= res["ci_upper"]

def test_bootstrap_confidence_interval_width():
    sample = np.random.default_rng(42).normal(loc=50.0, scale=10.0, size=50)
    boot = BootstrapEstimator(sample, seed=42)
    
    res_90 = boot.estimate(statistic_func=np.mean, n_iterations=2_000, ci_level=0.90)
    res_99 = boot.estimate(statistic_func=np.mean, n_iterations=2_000, ci_level=0.99)
    
    width_90 = res_90["ci_upper"] - res_90["ci_lower"]
    width_99 = res_99["ci_upper"] - res_99["ci_lower"]
    
    # 99% CI must be wider than 90% CI
    assert width_99 > width_90
