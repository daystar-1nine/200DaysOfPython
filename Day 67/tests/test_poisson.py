"""
Unit tests for Poisson distribution.
"""
import pytest
from distributions.poisson import PoissonDistribution

def test_poisson_pmf(poisson_dist):
    # lambda = 5.0, P(X=5) = e^-5 * 5^5 / 5! = 0.175467
    p5 = poisson_dist.pmf_or_pdf(5)
    assert p5 == pytest.approx(0.175467, abs=1e-4)

def test_poisson_cdf(poisson_dist):
    assert poisson_dist.cdf(5) == pytest.approx(0.615961, abs=1e-4)

def test_poisson_mean_and_variance(poisson_dist):
    assert poisson_dist.mean() == pytest.approx(5.0)
    assert poisson_dist.variance() == pytest.approx(5.0)
    assert poisson_dist.std() == pytest.approx(5.0 ** 0.5)

def test_poisson_ppf(poisson_dist):
    median_val = poisson_dist.ppf(0.50)
    assert median_val in [4.0, 5.0]

def test_poisson_invalid_rate():
    with pytest.raises(ValueError, match="non-negative"):
        PoissonDistribution(lam=-1.0)
