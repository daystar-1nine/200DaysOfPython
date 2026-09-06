"""
Unit tests for basic probability functions.
"""
import pytest
from app.probability.basic import (
    theoretical_probability,
    empirical_probability,
    complement_probability,
    union_probability
)

def test_theoretical_probability_valid():
    assert theoretical_probability(1, 6) == pytest.approx(1/6)
    assert theoretical_probability(0, 10) == 0.0
    assert theoretical_probability(10, 10) == 1.0

def test_theoretical_probability_invalid_sample_space():
    with pytest.raises(ValueError, match="strictly positive"):
        theoretical_probability(1, 0)
    with pytest.raises(ValueError, match="strictly positive"):
        theoretical_probability(1, -5)

def test_theoretical_probability_invalid_favorable():
    with pytest.raises(ValueError, match="between 0 and sample_space_size"):
        theoretical_probability(-1, 6)
    with pytest.raises(ValueError, match="between 0 and sample_space_size"):
        theoretical_probability(7, 6)

def test_empirical_probability_valid():
    assert empirical_probability(48, 100) == 0.48
    assert empirical_probability(0, 50) == 0.0
    assert empirical_probability(50, 50) == 1.0

def test_empirical_probability_invalid():
    with pytest.raises(ValueError, match="strictly positive"):
        empirical_probability(5, 0)
    with pytest.raises(ValueError, match="between 0 and total_trials"):
        empirical_probability(-2, 10)
    with pytest.raises(ValueError, match="between 0 and total_trials"):
        empirical_probability(15, 10)

def test_complement_probability():
    assert complement_probability(0.3) == pytest.approx(0.7)
    assert complement_probability(0.0) == 1.0
    assert complement_probability(1.0) == 0.0

def test_complement_probability_out_of_bounds():
    with pytest.raises(ValueError):
        complement_probability(1.2)
    with pytest.raises(ValueError):
        complement_probability(-0.1)

def test_union_probability_disjoint():
    # P(A) = 0.2, P(B) = 0.3, mutually exclusive -> P(A or B) = 0.5
    assert union_probability(0.2, 0.3, 0.0) == pytest.approx(0.5)

def test_union_probability_overlapping():
    # P(A) = 0.5, P(B) = 0.4, P(A and B) = 0.2 -> P(A or B) = 0.7
    assert union_probability(0.5, 0.4, 0.2) == pytest.approx(0.7)

def test_union_probability_invalid_intersection():
    with pytest.raises(ValueError, match="cannot exceed"):
        union_probability(0.3, 0.4, 0.5)
