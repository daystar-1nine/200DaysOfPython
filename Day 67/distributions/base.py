"""
Abstract Base Distribution Interface for Probability Distribution Analyzer.
"""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseDistribution(ABC):
    """Abstract interface defining standard probability distribution capabilities."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the probability distribution."""
        pass
        
    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """True if the distribution is discrete, False if continuous."""
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Dictionary of parameter names and values."""
        pass

    @abstractmethod
    def pmf_or_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate PMF (if discrete) or PDF (if continuous) at point(s) x."""
        pass

    @abstractmethod
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Cumulative Distribution Function P(X <= x)."""
        pass

    @abstractmethod
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        """Percent Point Function (Quantile / Inverse CDF) for cumulative probability q."""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Theoretical expected value E[X]."""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Theoretical variance Var(X)."""
        pass

    def std(self) -> float:
        """Theoretical standard deviation sqrt(Var(X))."""
        return float(np.sqrt(self.variance()))

    @abstractmethod
    def sample(self, size: int = 1000, random_state: int | None = 42) -> np.ndarray:
        """Generate random variates drawn from the distribution."""
        pass

    def get_summary(self) -> dict[str, Any]:
        """Return standardized theoretical summary dictionary."""
        return {
            "distribution": self.name,
            "is_discrete": self.is_discrete,
            "parameters": self.parameters,
            "mean": round(self.mean(), 6),
            "variance": round(self.variance(), 6),
            "std": round(self.std(), 6)
        }
