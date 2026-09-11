"""
Unit tests for confidence interval functions.
"""

import math
import pytest

try:
    from app.confidence_intervals import mean_difference_ci, proportion_difference_ci
except (ImportError, ModuleNotFoundError):
    from confidence_intervals import mean_difference_ci, proportion_difference_ci

def test_mean_difference_ci():
    a = [10.0, 12.0, 11.0, 13.0, 10.5]
    b = [15.0, 16.0, 17.0, 14.5, 16.5]
    ci = mean_difference_ci(a, b, confidence=0.95)
    assert ci["difference"] > 0
    assert ci["ci_lower"] < ci["difference"] < ci["ci_upper"]
    assert ci["excludes_zero"] is True

def test_proportion_difference_ci():
    ci = proportion_difference_ci(p_control=0.05, n_control=10000, p_treatment=0.06, n_treatment=10000, confidence=0.95)
    assert math.isclose(ci["difference"], 0.01)
    assert ci["ci_lower"] < 0.01 < ci["ci_upper"]
    assert ci["excludes_zero"] is True
