"""
Bernoulli Distribution Module.
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

class BernoulliDistribution(BaseDistribution):
    """Bernoulli distribution representing a single binary trial with success probability p."""
    
    def __init__(self, p: float):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Bernoulli probability p must be in [0, 1], got {p}.")
        self._p = float(p)
        self._rv = stats.bernoulli(self._p)
        
    @property
    def name(self) -> str:
        return "Bernoulli"
        
    @property
    def is_discrete(self) -> bool:
        return True
        
    @property
    def parameters(self) -> dict[str, Any]:
        return {"p": self._p}
        
    def pmf_or_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.pmf(x)
        
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.cdf(x)
        
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._rv.ppf(q)
        
    def mean(self) -> float:
        return self._p
        
    def variance(self) -> float:
        return self._p * (1.0 - self._p)
        
    def sample(self, size: int = 1000, random_state: int | None = 42) -> np.ndarray:
        return self._rv.rvs(size=size, random_state=random_state)
