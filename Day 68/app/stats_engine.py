"""
Core statistical functions for sampling and inference.
"""
import numpy as np
from scipy import stats

def compute_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))

def compute_variance(arr: np.ndarray, ddof: int = 0) -> float:
    return float(np.var(arr, ddof=ddof))

def compute_std(arr: np.ndarray, ddof: int = 0) -> float:
    return float(np.std(arr, ddof=ddof))

def compute_theoretical_se(pop_std: float, sample_size: int, finite_pop_size: int | None = None) -> float:
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    base_se = pop_std / np.sqrt(sample_size)
    if finite_pop_size is not None and finite_pop_size > 1:
        if sample_size > finite_pop_size:
            raise ValueError("Sample size cannot exceed population size.")
        fpc = np.sqrt((finite_pop_size - sample_size) / (finite_pop_size - 1))
        return float(base_se * fpc)
    return float(base_se)

def compute_skewness(arr: np.ndarray) -> float:
    return float(stats.skew(arr))

def compute_kurtosis(arr: np.ndarray) -> float:
    return float(stats.kurtosis(arr))
