"""
Continuous Uniform Distribution Module.
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

class UniformDistribution(BaseDistribution):
    """Continuous Uniform distribution over the interval [a, b]."""
    
    def __init__(self, a: float, b: float):
        if b <= a:
            raise ValueError(f"Uniform upper bound b ({b}) must be strictly greater than lower bound a ({a}).")
        self._a = float(a)
        self._b = float(b)
        self._rv = stats.uniform(loc=self._a, scale=self._b - self._a)
        
    @property
    def name(self) -> str:
        return "Uniform"
        
    @property
    def is_discrete(self) -> bool:
        return False
        
    @property
    def parameters(self) -> dict[str, Any]:
        return {"a": self._a, "b": self._b}
        
    def pmf_or_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.pdf(x)
        
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.cdf(x)
        
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._rv.ppf(q)
        
    def mean(self) -> float:
        return (self._a + self._b) / 2.0
        
    def variance(self) -> float:
        return ((self._b - self._a) ** 2) / 12.0
        
    def sample(self, size: int = 1000, random_state: int | None = 42) -> np.ndarray:
        return self._rv.rvs(size=size, random_state=random_state)
