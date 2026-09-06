"""
Unit tests for Normal distribution.
"""
import pytest
import numpy as np
from distributions.normal import NormalDistribution

def test_normal_pdf(normal_dist):
    # Peak at mean
    peak_pdf = normal_dist.pmf_or_pdf(70.0)
    assert peak_pdf == pytest.approx(1.0 / (10.0 * np.sqrt(2 * np.pi)), abs=1e-4)

def test_normal_cdf(normal_dist):
    # Symmetrical around 70
    assert normal_dist.cdf(70.0) == pytest.approx(0.5)
    # Below 1 sigma
    assert normal_dist.cdf(60.0) == pytest.approx(0.158655, abs=1e-4)

def test_normal_ppf(normal_dist):
    assert normal_dist.ppf(0.5) == pytest.approx(70.0)
    # 97.5% quantile is approx mu + 1.96 * sigma = 70 + 19.6 = 89.6
    assert normal_dist.ppf(0.975) == pytest.approx(89.5996, abs=1e-2)

def test_normal_empirical_rule(normal_dist):
    # 68-95-99.7 rule
    p_1s = normal_dist.cdf(80.0) - normal_dist.cdf(60.0)
    p_2s = normal_dist.cdf(90.0) - normal_dist.cdf(50.0)
    p_3s = normal_dist.cdf(100.0) - normal_dist.cdf(40.0)
    assert p_1s == pytest.approx(0.6827, abs=1e-3)
    assert p_2s == pytest.approx(0.9545, abs=1e-3)
    assert p_3s == pytest.approx(0.9973, abs=1e-3)

def test_normal_mean_and_variance(normal_dist):
    assert normal_dist.mean() == pytest.approx(70.0)
    assert normal_dist.variance() == pytest.approx(100.0)
    assert normal_dist.std() == pytest.approx(10.0)

def test_normal_invalid_sigma():
    with pytest.raises(ValueError, match="strictly positive"):
        NormalDistribution(mu=0.0, sigma=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        NormalDistribution(mu=0.0, sigma=-2.0)
