"""
Unit tests for expected value and variance of random variables.
"""
import pytest
from app.probability.expected_value import calculate_expected_value, calculate_variance_discrete

def test_calculate_expected_value_die(fair_die_values, fair_die_probs):
    ev = calculate_expected_value(fair_die_values, fair_die_probs)
    assert ev == pytest.approx(3.5)

def test_calculate_expected_value_coin():
    ev = calculate_expected_value([0, 1], [0.5, 0.5])
    assert ev == pytest.approx(0.5)

def test_calculate_expected_value_mismatched_lengths():
    with pytest.raises(ValueError, match="identical lengths"):
        calculate_expected_value([1, 2], [0.5])

def test_calculate_expected_value_invalid_sum():
    with pytest.raises(ValueError, match="sum to 1.0"):
        calculate_expected_value([1, 2], [0.4, 0.4])

def test_calculate_variance_discrete(fair_die_values, fair_die_probs):
    var_res = calculate_variance_discrete(fair_die_values, fair_die_probs)
    assert var_res["expected_value"] == pytest.approx(3.5)
    # Var of fair 6-sided die is (6^2 - 1) / 12 = 35 / 12 = 2.9167
    assert var_res["variance"] == pytest.approx(35 / 12, abs=1e-3)
    assert var_res["std_dev"] == pytest.approx((35 / 12)**0.5, abs=1e-3)
