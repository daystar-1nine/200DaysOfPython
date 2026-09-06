"""
Unit tests for conditional probability and independence.
"""
import pytest
from app.probability.conditional import (
    conditional_probability,
    joint_probability,
    independence_test
)

def test_conditional_probability_standard():
    # P(A and B) = 0.15, P(B) = 0.30 -> P(A | B) = 0.5
    assert conditional_probability(0.15, 0.30) == pytest.approx(0.50)

def test_conditional_probability_zero_or_negative_denominator():
    with pytest.raises(ValueError):
        conditional_probability(0.1, 0.0)
    with pytest.raises(ValueError):
        conditional_probability(0.1, -0.2)

def test_conditional_probability_numerator_greater():
    with pytest.raises(ValueError, match="cannot exceed"):
        conditional_probability(0.6, 0.4)

def test_joint_probability():
    assert joint_probability(0.4, 0.5) == pytest.approx(0.20)
    assert joint_probability(0.0, 0.5) == 0.0

def test_independence_test_independent():
    # P(A) = 0.5, P(B) = 0.4, P(A and B) = 0.20 -> independent
    res = independence_test(0.5, 0.4, 0.20)
    assert res["is_independent"] is True
    assert res["absolute_difference"] == pytest.approx(0.0)

def test_independence_test_dependent():
    # P(A) = 0.5, P(B) = 0.4, P(A and B) = 0.30 -> dependent
    res = independence_test(0.5, 0.4, 0.30)
    assert res["is_independent"] is False
    assert res["absolute_difference"] == pytest.approx(0.10)
