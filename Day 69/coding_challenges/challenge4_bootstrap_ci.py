"""
Day 69 Challenge 4: Non-Parametric Bootstrap Confidence Interval
Estimates confidence intervals for arbitrary statistics using empirical percentile resampling.
"""
from typing import Callable
import numpy as np

def bootstrap_ci(
    data: np.ndarray | list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Perform non-parametric bootstrap resampling with replacement and return percentile CI.
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        raise ValueError("Data array cannot be empty.")
    if iterations <= 0:
        raise ValueError("Iterations must be strictly positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence must be strictly between 0 and 1.")
        
    rng = np.random.default_rng(seed)
    n = len(arr)
    
    boot_stats = np.empty(iterations, dtype=float)
    for i in range(iterations):
        resample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic(resample)
        
    obs_val = float(statistic(arr))
    boot_mean = float(np.mean(boot_stats))
    boot_se = float(np.std(boot_stats, ddof=1))
    
    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    
    lower_bound = float(np.percentile(boot_stats, lower_pct))
    upper_bound = float(np.percentile(boot_stats, upper_pct))
    
    return {
        "sample_size": n,
        "iterations": iterations,
        "observed_statistic": round(obs_val, 4),
        "bootstrap_mean": round(boot_mean, 4),
        "bootstrap_se": round(boot_se, 4),
        "confidence_level": confidence,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "distribution": boot_stats
    }

def main():
    print("=" * 70)
    print("  CHALLENGE 4: NON-PARAMETRIC BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)
    # Right-skewed sample (customer transaction value)
    data = [120.0, 145.0, 130.0, 160.0, 125.0, 180.0, 135.0, 190.0, 350.0, 520.0]
    
    # 1. Bootstrap Mean
    res_mean = bootstrap_ci(data, statistic=np.mean, iterations=10_000, confidence=0.95)
    print("[1. Bootstrap Mean (95% CI)]")
    print(f"Observed: {res_mean['observed_statistic']:.2f} | SE: {res_mean['bootstrap_se']:.4f}")
    print(f"95% Percentile CI: [{res_mean['lower_bound']:.2f}, {res_mean['upper_bound']:.2f}]")
    
    # 2. Bootstrap Median
    res_med = bootstrap_ci(data, statistic=np.median, iterations=10_000, confidence=0.95)
    print("\n[2. Bootstrap Median (95% CI)]")
    print(f"Observed: {res_med['observed_statistic']:.2f} | SE: {res_med['bootstrap_se']:.4f}")
    print(f"95% Percentile CI: [{res_med['lower_bound']:.2f}, {res_med['upper_bound']:.2f}]")

if __name__ == "__main__":
    main()
