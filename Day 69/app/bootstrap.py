"""
Non-parametric bootstrap resampling engine.
"""
from typing import Callable
import numpy as np
try:
    from app.validator import validate_numeric_array, validate_confidence_level
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array, validate_confidence_level

class BootstrapEstimator:
    def __init__(self, data: np.ndarray, seed: int = 42):
        self.data = validate_numeric_array(data)
        self.rng = np.random.default_rng(seed)

    def estimate(
        self,
        statistic: Callable[[np.ndarray], float] = np.mean,
        iterations: int = 10_000,
        confidence: float = 0.95
    ) -> dict:
        if iterations <= 0:
            raise ValueError("Bootstrap iterations must be strictly positive.")
        conf = validate_confidence_level(confidence)
        
        n = len(self.data)
        boot_stats = np.empty(iterations, dtype=float)
        
        for i in range(iterations):
            resample = self.rng.choice(self.data, size=n, replace=True)
            boot_stats[i] = statistic(resample)
            
        obs_val = float(statistic(self.data))
        boot_mean = float(np.mean(boot_stats))
        boot_se = float(np.std(boot_stats, ddof=1))
        
        alpha = 1.0 - conf
        lower_bound = float(np.percentile(boot_stats, (alpha / 2.0) * 100.0))
        upper_bound = float(np.percentile(boot_stats, (1.0 - alpha / 2.0) * 100.0))
        
        return {
            "sample_size": n,
            "iterations": iterations,
            "observed_statistic": round(obs_val, 4),
            "bootstrap_mean": round(boot_mean, 4),
            "bootstrap_se": round(boot_se, 4),
            "confidence_level": conf,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "interval_width": round(upper_bound - lower_bound, 4),
            "distribution": boot_stats
        }
