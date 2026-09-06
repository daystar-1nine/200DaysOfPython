"""
Poisson Distribution Module.
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

class PoissonDistribution(BaseDistribution):
    """Discrete Poisson distribution parameterized by event arrival rate lambda (mu)."""
    
    def __init__(self, lam: float):
        if lam < 0.0:
            raise ValueError(f"Poisson rate lambda must be non-negative (>=0), got {lam}.")
        self._lam = float(lam)
        self._rv = stats.poisson(mu=self._lam)
        
    @property
    def name(self) -> str:
        return "Poisson"
        
    @property
    def is_discrete(self) -> bool:
        return True
        
    @property
    def parameters(self) -> dict[str, Any]:
        return {"lambda": self._lam}
        
    def pmf_or_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.pmf(x)
        
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.cdf(x)
        
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._rv.ppf(q)
        
    def mean(self) -> float:
        return self._lam
        
    def variance(self) -> float:
        return self._lam
        
    def sample(self, size: int = 1000, random_state: int | None = 42) -> np.ndarray:
        return self._rv.rvs(size=size, random_state=random_state)
