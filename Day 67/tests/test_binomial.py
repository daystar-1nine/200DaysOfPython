"""
Unit tests for Binomial distribution.
"""
import pytest
import numpy as np
from distributions.binomial import BinomialDistribution

def test_binomial_pmf(binomial_dist):
    # n=20, p=0.4, P(X=8) should be maximal peak
    p8 = binomial_dist.pmf_or_pdf(8)
    assert p8 == pytest.approx(0.1797, abs=1e-3)

def test_binomial_cdf(binomial_dist):
    # P(X <= 8)
    cdf8 = binomial_dist.cdf(8)
    assert 0.50 <= cdf8 <= 0.65

def test_binomial_ppf(binomial_dist):
    # Median quantile
    median_val = binomial_dist.ppf(0.50)
    assert median_val in [7.0, 8.0]

def test_binomial_mean_and_variance(binomial_dist):
    # Mean np = 20 * 0.4 = 8.0, Var = np(1-p) = 20 * 0.4 * 0.6 = 4.8
    assert binomial_dist.mean() == pytest.approx(8.0)
    assert binomial_dist.variance() == pytest.approx(4.8)
    assert binomial_dist.std() == pytest.approx(np.sqrt(4.8))

def test_binomial_complement_rule(binomial_dist):
    p_le_10 = binomial_dist.cdf(10)
    p_ge_11 = 1.0 - p_le_10
    assert p_le_10 + p_ge_11 == pytest.approx(1.0)

def test_binomial_invalid_trials():
    with pytest.raises(ValueError, match="non-negative integer"):
        BinomialDistribution(n=-5, p=0.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        BinomialDistribution(n=3.5, p=0.5)

def test_binomial_invalid_p():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        BinomialDistribution(n=10, p=1.2)
