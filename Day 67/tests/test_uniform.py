"""
Unit tests for Uniform distribution.
"""
import pytest
import numpy as np
from distributions.uniform import UniformDistribution

def test_uniform_pdf(uniform_dist):
    # a=0, b=10 -> PDF = 1/10 = 0.1
    assert uniform_dist.pmf_or_pdf(5.0) == pytest.approx(0.1)
    assert uniform_dist.pmf_or_pdf(-1.0) == pytest.approx(0.0)
    assert uniform_dist.pmf_or_pdf(11.0) == pytest.approx(0.0)

def test_uniform_cdf(uniform_dist):
    assert uniform_dist.cdf(5.0) == pytest.approx(0.5)
    assert uniform_dist.cdf(0.0) == pytest.approx(0.0)
    assert uniform_dist.cdf(10.0) == pytest.approx(1.0)

def test_uniform_ppf(uniform_dist):
    assert uniform_dist.ppf(0.5) == pytest.approx(5.0)
    assert uniform_dist.ppf(0.9) == pytest.approx(9.0)

def test_uniform_mean_and_variance(uniform_dist):
    # Mean = (0+10)/2 = 5.0, Var = (10-0)^2 / 12 = 100 / 12 = 8.3333
    assert uniform_dist.mean() == pytest.approx(5.0)
    assert uniform_dist.variance() == pytest.approx(100 / 12)

def test_uniform_sampling_bounds(uniform_dist):
    samples = uniform_dist.sample(size=1000, random_state=42)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 10.0)

def test_uniform_invalid_bounds():
    with pytest.raises(ValueError, match="strictly greater"):
        UniformDistribution(a=10.0, b=5.0)
    with pytest.raises(ValueError, match="strictly greater"):
        UniformDistribution(a=5.0, b=5.0)
