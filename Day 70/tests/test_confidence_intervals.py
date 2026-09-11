"""
Unit tests for confidence interval calculations and duality checks.
"""

import math
import pytest

try:
    from app.confidence_intervals import compute_mean_ci, check_ci_contains_null
except (ImportError, ModuleNotFoundError):
    from confidence_intervals import compute_mean_ci, check_ci_contains_null

def test_compute_mean_ci():
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    ci = compute_mean_ci(data, confidence_level=0.95)
    assert ci["confidence_level"] == 0.95
    assert ci["mean"] == 14.0
    assert ci["ci_lower"] < 14.0 < ci["ci_upper"]
    assert math.isclose(ci["ci_upper"] - ci["mean"], ci["margin_of_error"])

def test_check_ci_contains_null():
    assert check_ci_contains_null(ci_lower=10.0, ci_upper=20.0, null_val=15.0) is True
    assert check_ci_contains_null(ci_lower=10.0, ci_upper=20.0, null_val=10.0) is True
    assert check_ci_contains_null(ci_lower=10.0, ci_upper=20.0, null_val=20.0) is True
    assert check_ci_contains_null(ci_lower=10.0, ci_upper=20.0, null_val=25.0) is False
    assert check_ci_contains_null(ci_lower=10.0, ci_upper=20.0, null_val=5.0) is False
