"""
Day 68 Coding Challenge 4: Universal Bootstrap Resampling Engine
Implements bootstrap_statistic() capable of estimating uncertainty and
percentile confidence intervals for arbitrary functions (mean, median, std, etc.).
"""
from typing import Callable
import numpy as np

def bootstrap_statistic(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Perform non-parametric bootstrap resampling for an arbitrary statistic function.
    Returns observed statistic, bootstrap mean, standard error, and percentile confidence interval.
    """
    if len(data) == 0:
        raise ValueError("Data array cannot be empty.")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be strictly positive.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1.")
        
    rng = np.random.default_rng(seed)
    n = len(data)
    
    boot_stats = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_func(resample)
        
    obs_val = float(statistic_func(data))
    boot_mean = float(np.mean(boot_stats))
    boot_se = float(np.std(boot_stats, ddof=1))
    
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1.0 - alpha) * 100))
    
    return {
        "observed_statistic": round(obs_val, 4),
        "bootstrap_mean": round(boot_mean, 4),
        "bootstrap_se": round(boot_se, 4),
        "ci_level": ci_level,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "bootstrap_distribution": boot_stats
    }

def main():
    print("=" * 70)
    print("  CHALLENGE 4: UNIVERSAL BOOTSTRAP RESAMPLING ENGINE")
    print("=" * 70)
    # Right-skewed sample data (income or order values)
    sample_data = np.array([22.5, 28.0, 31.5, 34.0, 39.0, 42.0, 45.0, 58.0, 95.0, 150.0])
    
    # 1. Bootstrap Mean
    res_mean = bootstrap_statistic(sample_data, np.mean, n_iterations=10_000)
    print("[1. Bootstrap Mean]")
    print(f"Observed Mean: {res_mean['observed_statistic']:.2f} | Bootstrap SE: {res_mean['bootstrap_se']:.4f}")
    print(f"95% CI: [{res_mean['ci_lower']:.2f}, {res_mean['ci_upper']:.2f}]")
    
    # 2. Bootstrap Median
    res_med = bootstrap_statistic(sample_data, np.median, n_iterations=10_000)
    print("\n[2. Bootstrap Median]")
    print(f"Observed Median: {res_med['observed_statistic']:.2f} | Bootstrap SE: {res_med['bootstrap_se']:.4f}")
    print(f"95% CI: [{res_med['ci_lower']:.2f}, {res_med['ci_upper']:.2f}]")

if __name__ == "__main__":
    main()
