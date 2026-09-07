"""
Point and parameter estimation abstractions.
"""
import numpy as np
try:
    from app.validator import validate_numeric_array, validate_proportion_counts
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array, validate_proportion_counts

class PointEstimator:
    @staticmethod
    def estimate_mean(data: np.ndarray) -> float:
        arr = validate_numeric_array(data)
        return float(np.mean(arr))

    @staticmethod
    def estimate_variance(data: np.ndarray, ddof: int = 1) -> float:
        arr = validate_numeric_array(data)
        if len(arr) < 2 and ddof == 1:
            raise ValueError("Sample variance requires at least 2 observations.")
        return float(np.var(arr, ddof=ddof))

    @staticmethod
    def estimate_std(data: np.ndarray, ddof: int = 1) -> float:
        return float(np.sqrt(PointEstimator.estimate_variance(data, ddof=ddof)))

    @staticmethod
    def estimate_proportion(successes: int, total: int) -> float:
        x, n = validate_proportion_counts(successes, total)
        return float(x / n)
