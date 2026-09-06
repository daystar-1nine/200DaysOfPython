"""
Non-parametric bootstrap estimation for single samples.
"""
from typing import Callable
import numpy as np

class BootstrapEstimator:
    def __init__(self, sample: np.ndarray, seed: int = 42):
        if len(sample) == 0:
            raise ValueError("Sample cannot be empty.")
        self.sample = sample
        self.rng = np.random.default_rng(seed)

    def estimate(
        self,
        statistic_func: Callable[[np.ndarray], float] = np.mean,
        n_iterations: int = 5_000,
        ci_level: float = 0.95
    ) -> dict:
        n = len(self.sample)
        boot_stats = np.empty(n_iterations, dtype=float)

        for i in range(n_iterations):
            resample = self.rng.choice(self.sample, size=n, replace=True)
            boot_stats[i] = statistic_func(resample)

        observed_stat = float(statistic_func(self.sample))
        boot_mean = float(np.mean(boot_stats))
        boot_se = float(np.std(boot_stats, ddof=1))

        alpha = (1.0 - ci_level) / 2.0
        ci_lower = float(np.percentile(boot_stats, alpha * 100))
        ci_upper = float(np.percentile(boot_stats, (1.0 - alpha) * 100))

        return {
            "observed": observed_stat,
            "boot_mean": boot_mean,
            "boot_se": boot_se,
            "ci_level": ci_level,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "distribution": boot_stats
        }
