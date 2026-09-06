"""
Tests for SamplingDistributionSimulator.
"""
import pytest
import numpy as np
from app.sampling_distribution import SamplingDistributionSimulator

def test_simulator_initialization(normal_population):
    sim = SamplingDistributionSimulator(normal_population, seed=42)
    assert np.isclose(sim.pop_mean, 100.0, atol=1.0)
    assert np.isclose(sim.pop_std, 15.0, atol=1.0)

def test_simulate_means_unbiasedness(normal_population):
    sim = SamplingDistributionSimulator(normal_population, seed=42)
    res = sim.simulate_means(sample_size=30, n_samples=2_000)
    
    # Expected value of sample mean equals population mean
    assert np.isclose(res["emp_mean"], res["pop_mean"], atol=0.5)

def test_simulate_means_standard_error_agreement(normal_population):
    sim = SamplingDistributionSimulator(normal_population, seed=42)
    res = sim.simulate_means(sample_size=25, n_samples=3_000)
    
    # Empirical standard error should be very close to theoretical standard error
    assert np.isclose(res["emp_se"], res["theo_se"], atol=0.2)
    assert res["se_difference"] < 0.2

def test_simulate_means_output_keys(skewed_population):
    sim = SamplingDistributionSimulator(skewed_population, seed=42)
    res = sim.simulate_means(sample_size=20, n_samples=500)
    
    expected_keys = {
        "sample_size", "n_samples", "means", "pop_mean", "pop_std",
        "emp_mean", "emp_se", "theo_se", "se_difference", "skewness"
    }
    assert expected_keys.issubset(set(res.keys()))
    assert len(res["means"]) == 500
