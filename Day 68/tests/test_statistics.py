"""
Tests for core statistical functions.
"""
import pytest
import numpy as np
from app.stats_engine import (
    compute_mean,
    compute_variance,
    compute_std,
    compute_theoretical_se,
    compute_skewness,
    compute_kurtosis
)

def test_compute_mean():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert compute_mean(data) == 30.0

def test_compute_variance():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert compute_variance(data, ddof=0) == 200.0
    assert compute_variance(data, ddof=1) == 250.0

def test_compute_std():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert np.isclose(compute_std(data, ddof=0), np.sqrt(200.0))
    assert np.isclose(compute_std(data, ddof=1), np.sqrt(250.0))

def test_compute_theoretical_se_infinite():
    pop_std = 15.0
    n = 25
    se = compute_theoretical_se(pop_std, n)
    assert se == 3.0  # 15 / sqrt(25) = 15 / 5 = 3.0

def test_compute_theoretical_se_fpc():
    pop_std = 15.0
    n = 25
    N = 100
    se_fpc = compute_theoretical_se(pop_std, n, finite_pop_size=N)
    fpc = np.sqrt((100 - 25) / (100 - 1))
    assert np.isclose(se_fpc, 3.0 * fpc)

def test_compute_theoretical_se_invalid_n():
    with pytest.raises(ValueError, match="Sample size must be positive"):
        compute_theoretical_se(10.0, 0)

def test_compute_theoretical_se_n_exceeds_N():
    with pytest.raises(ValueError, match="Sample size cannot exceed population size"):
        compute_theoretical_se(10.0, 50, finite_pop_size=30)

def test_compute_skewness_symmetric():
    data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    assert np.isclose(compute_skewness(data), 0.0, atol=1e-5)

def test_compute_skewness_right_skewed():
    data = np.array([1.0, 2.0, 2.0, 3.0, 100.0])
    assert compute_skewness(data) > 1.0

def test_compute_kurtosis():
    rng = np.random.default_rng(42)
    norm = rng.normal(loc=0, scale=1, size=10_000)
    # Fisher kurtosis of standard normal is close to 0
    assert np.isclose(compute_kurtosis(norm), 0.0, atol=0.2)
