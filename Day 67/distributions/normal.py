"""
Normal (Gaussian) Distribution Module.
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

class NormalDistribution(BaseDistribution):
    """Continuous Normal (Gaussian) distribution parameterized by mean mu and standard deviation sigma."""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0.0:
            raise ValueError(f"Normal standard deviation sigma must be strictly positive (>0), got {sigma}.")
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._rv = stats.norm(loc=self._mu, scale=self._sigma)
        
    @property
    def name(self) -> str:
        return "Normal"
        
    @property
    def is_discrete(self) -> bool:
        return False
        
    @property
    def parameters(self) -> dict[str, Any]:
        return {"mu": self._mu, "sigma": self._sigma}
        
    def pmf_or_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.pdf(x)
        
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._rv.cdf(x)
        
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._rv.ppf(q)
        
    def mean(self) -> float:
        return self._mu
        
    def variance(self) -> float:
        return self._sigma ** 2
        
    def sample(self, size: int = 1000, random_state: int | None = 42) -> np.ndarray:
        return self._rv.rvs(size=size, random_state=random_state)
