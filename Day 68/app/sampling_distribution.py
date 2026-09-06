"""
Sampling distribution simulator for sample statistics (mean, median, etc.).
"""
import numpy as np
try:
    from app.sampler import RandomSampler
    from app.stats_engine import compute_mean, compute_std, compute_theoretical_se, compute_skewness
except (ImportError, ModuleNotFoundError):
    from sampler import RandomSampler
    from stats_engine import compute_mean, compute_std, compute_theoretical_se, compute_skewness


class SamplingDistributionSimulator:
    def __init__(self, population: np.ndarray, seed: int = 42):
        self.population = population
        self.pop_mean = compute_mean(population)
        self.pop_std = compute_std(population, ddof=0)
        self.sampler = RandomSampler(seed=seed)

    def simulate_means(self, sample_size: int, n_samples: int = 5_000, replace: bool = False) -> dict:
        means = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            samp = self.sampler.sample(self.population, size=sample_size, replace=replace)
            means[i] = np.mean(samp)

        emp_mean = float(np.mean(means))
        emp_se = float(np.std(means, ddof=1))
        theo_se = compute_theoretical_se(self.pop_std, sample_size)
        skew = compute_skewness(means)

        return {
            "sample_size": sample_size,
            "n_samples": n_samples,
            "means": means,
            "pop_mean": self.pop_mean,
            "pop_std": self.pop_std,
            "emp_mean": emp_mean,
            "emp_se": emp_se,
            "theo_se": theo_se,
            "se_difference": abs(emp_se - theo_se),
            "skewness": skew
        }
