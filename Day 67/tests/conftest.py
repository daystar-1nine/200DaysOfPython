"""
Shared fixtures and test configuration for Day 67.
"""
import sys
from pathlib import Path
import pytest

day67_dir = Path(__file__).resolve().parent.parent
if str(day67_dir) not in sys.path:
    sys.path.insert(0, str(day67_dir))

from distributions.bernoulli import BernoulliDistribution
from distributions.binomial import BinomialDistribution
from distributions.uniform import UniformDistribution
from distributions.normal import NormalDistribution
from distributions.poisson import PoissonDistribution

@pytest.fixture
def bernoulli_dist():
    return BernoulliDistribution(p=0.7)

@pytest.fixture
def binomial_dist():
    return BinomialDistribution(n=20, p=0.4)

@pytest.fixture
def uniform_dist():
    return UniformDistribution(a=0.0, b=10.0)

@pytest.fixture
def normal_dist():
    return NormalDistribution(mu=70.0, sigma=10.0)

@pytest.fixture
def poisson_dist():
    return PoissonDistribution(lam=5.0)
