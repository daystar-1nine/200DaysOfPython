"""
Binomial Distribution Module.
"""
import sys
from pathlib import Path
day67_dir = Path(__file__).resolve().parent.parent
if str(day67_dir) not in sys.path:
    sys.path.insert(0, str(day67_dir))

from typing import Any
import numpy as np
from scipy import stats

try:
    from .base import BaseDistribution
except ImportError:
    from distributions.base import BaseDistribution

class BinomialDistribution(BaseDistribution):
    """Binomial distribution representing number of successes in n independent Bernoulli trials."""
    
    def __init__(self, n: int, p: float):
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"Binomial trials n must be a non-negative integer, got {n}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Binomial success probability p must be in [0, 1], got {p}.")
        self._n = int(n)
        self._p = float(p)
        self._rv = stats.binom(self._n, self._p)
        
    @property
    def name(self) -> str:
        return "Binomial"
        
    @property
    def is_discrete(self) -> bool:
        return True
        
    @property
    def parameters(self) -> dict[str, Any]:
        return {"n": self._n, "p": self._p}
        
    def pmf_or_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.pmf(x)
        
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.cdf(x)
        
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._rv.ppf(q)
        
    def mean(self) -> float:
        return float(self._n * self._p)
        
    def variance(self) -> float:
        return float(self._n * self._p * (1.0 - self._p))
        
    def sample(self, size: int = 1000, random_state: int | None = 42) -> np.ndarray:
        return self._rv.rvs(size=size, random_state=random_state)
