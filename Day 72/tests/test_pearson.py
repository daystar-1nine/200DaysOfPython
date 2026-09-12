"""
Unit tests for Pearson correlation engine.
"""
import pytest
import numpy as np
from app.pearson import compute_pearson_detail

def test_pearson_perfect_positive(clean_linear_arrays):
    x, y = clean_linear_arrays
    res = compute_pearson_detail(x, y)
    assert np.isclose(res["r"], 1.0)
    assert res["p_value"] == 0.0
    assert res["direction"] == "Positive"
    assert "Very Strong" in res["strength"]
    assert res["is_significant"] is True

def test_pearson_perfect_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    res = compute_pearson_detail(x, y)
    assert np.isclose(res["r"], -1.0)
    assert res["direction"] == "Negative"
    assert "Very Strong" in res["strength"]

def test_pearson_fisher_ci_bounds():
    np.random.seed(42)
    x = np.linspace(1, 50, 50)
    y = 1.5 * x + np.random.normal(0, 10, 50)
    res = compute_pearson_detail(x, y, alpha=0.05)
    assert res["ci_lower"] < res["r"] < res["ci_upper"]
    assert -1.0 <= res["ci_lower"] <= 1.0
    assert -1.0 <= res["ci_upper"] <= 1.0

def test_pearson_insufficient_samples():
    with pytest.raises(ValueError, match="at least 3"):
        compute_pearson_detail(np.array([1, 2]), np.array([3, 4]))

def test_pearson_near_zero_linear():
    x = np.linspace(-10, 10, 101)
    y = x ** 2
    res = compute_pearson_detail(x, y)
    assert abs(res["r"]) < 0.05
    assert res["is_significant"] is False

def test_pearson_invariance_to_positive_linear_scaling():
    x = np.array([10, 20, 30, 40, 50], dtype=float)
    y = np.array([15, 28, 42, 59, 73], dtype=float)
    r_orig = compute_pearson_detail(x, y)["r"]
    # Scale and shift
    x_scaled = 5.0 * x + 100.0
    y_scaled = 2.5 * y - 20.0
    r_scaled = compute_pearson_detail(x_scaled, y_scaled)["r"]
    assert np.isclose(r_orig, r_scaled)

def test_pearson_sign_flip_on_negative_scaling():
    x = np.array([10, 20, 30, 40, 50], dtype=float)
    y = np.array([15, 28, 42, 59, 73], dtype=float)
    r_orig = compute_pearson_detail(x, y)["r"]
    x_neg = -1.0 * x
    r_neg = compute_pearson_detail(x_neg, y)["r"]
    assert np.isclose(r_orig, -r_neg)
