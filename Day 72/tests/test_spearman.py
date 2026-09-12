"""
Unit tests for Spearman rank correlation engine.
"""
import pytest
import numpy as np
from app.spearman import compute_spearman_detail

def test_spearman_monotonic_nonlinear():
    # Strict exponential curve: non-linear but perfectly monotonic
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = np.exp(x)
    res = compute_spearman_detail(x, y)
    assert np.isclose(res["rho"], 1.0)
    assert res["is_significant"] is True
    assert res["direction"] == "Positive"

def test_spearman_reverse_rank():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([50, 40, 30, 20, 10], dtype=float)
    res = compute_spearman_detail(x, y)
    assert np.isclose(res["rho"], -1.0)
    assert res["direction"] == "Negative"

def test_spearman_tied_ranks():
    x = np.array([1, 2, 2, 4, 5], dtype=float)
    y = np.array([10, 20, 20, 40, 50], dtype=float)
    res = compute_spearman_detail(x, y)
    assert np.isclose(res["rho"], 1.0)

def test_spearman_insufficient_samples():
    with pytest.raises(ValueError, match="at least 3"):
        compute_spearman_detail(np.array([1, 2]), np.array([3, 4]))

def test_spearman_monotonic_transformation_invariance():
    x = np.linspace(1, 20, 20)
    y = 3.0 * x + 5.0
    rho_orig = compute_spearman_detail(x, y)["rho"]
    # Apply strictly increasing non-linear logarithmic transformation
    y_log = np.log(y)
    rho_trans = compute_spearman_detail(x, y_log)["rho"]
    assert np.isclose(rho_orig, rho_trans)

def test_spearman_near_zero_random():
    np.random.seed(123)
    x = np.random.normal(0, 1, 500)
    y = np.random.normal(0, 1, 500)
    res = compute_spearman_detail(x, y)
    assert abs(res["rho"]) < 0.15
