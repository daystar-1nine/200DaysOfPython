"""
Probability Distributions Package for Day 67.
"""
try:
    from .base import BaseDistribution
    from .bernoulli import BernoulliDistribution
    from .binomial import BinomialDistribution
    from .uniform import UniformDistribution
    from .normal import NormalDistribution
    from .poisson import PoissonDistribution
except ImportError:
    from distributions.base import BaseDistribution
    from distributions.bernoulli import BernoulliDistribution
    from distributions.binomial import BinomialDistribution
    from distributions.uniform import UniformDistribution
    from distributions.normal import NormalDistribution
    from distributions.poisson import PoissonDistribution

__all__ = [
    "BaseDistribution",
    "BernoulliDistribution",
    "BinomialDistribution",
    "UniformDistribution",
    "NormalDistribution",
    "PoissonDistribution",
]
