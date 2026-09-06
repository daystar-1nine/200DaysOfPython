"""
Central Limit Theorem verification and convergence analysis.
"""
import numpy as np
import pandas as pd
from scipy import stats
try:
    from app.sampling_distribution import SamplingDistributionSimulator
except (ImportError, ModuleNotFoundError):
    from sampling_distribution import SamplingDistributionSimulator


class CLTAnalyzer:
    def __init__(self, population: np.ndarray, seed: int = 42):
        self.population = population
        self.simulator = SamplingDistributionSimulator(population, seed=seed)

    def evaluate_convergence(self, sample_sizes: list[int] | tuple[int, ...], n_samples: int = 5_000) -> pd.DataFrame:
        results = []
        for n in sample_sizes:
            sim = self.simulator.simulate_means(sample_size=n, n_samples=n_samples)
            means = sim["means"]
            _, p_value = stats.normaltest(means)
            
            results.append({
                "sample_size": n,
                "emp_mean": sim["emp_mean"],
                "theo_se": sim["theo_se"],
                "emp_se": sim["emp_se"],
                "se_error": abs(sim["emp_se"] - sim["theo_se"]),
                "skewness": sim["skewness"],
                "p_normaltest": float(p_value),
                "is_approx_normal": bool(abs(sim["skewness"]) < 0.15)
            })
        return pd.DataFrame(results)
