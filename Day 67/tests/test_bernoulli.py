"""
Unit tests for Bernoulli distribution.
"""
import pytest
import numpy as np
from distributions.bernoulli import BernoulliDistribution

def test_bernoulli_pmf_success(bernoulli_dist):
    assert bernoulli_dist.pmf_or_pdf(1) == pytest.approx(0.7)

def test_bernoulli_pmf_failure(bernoulli_dist):
    assert bernoulli_dist.pmf_or_pdf(0) == pytest.approx(0.3)

def test_bernoulli_mean(bernoulli_dist):
    assert bernoulli_dist.mean() == pytest.approx(0.7)

def test_bernoulli_variance(bernoulli_dist):
    assert bernoulli_dist.variance() == pytest.approx(0.21)

def test_bernoulli_std(bernoulli_dist):
    assert bernoulli_dist.std() == pytest.approx(np.sqrt(0.21))

def test_bernoulli_cdf(bernoulli_dist):
    assert bernoulli_dist.cdf(0) == pytest.approx(0.3)
    assert bernoulli_dist.cdf(1) == pytest.approx(1.0)

def test_bernoulli_sampling(bernoulli_dist):
    samples = bernoulli_dist.sample(size=1000, random_state=42)
    assert len(samples) == 1000
    assert set(np.unique(samples)).issubset({0, 1})

def test_bernoulli_invalid_probability():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        BernoulliDistribution(p=1.5)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        BernoulliDistribution(p=-0.1)
