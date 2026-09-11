"""
Task 2: One-Sample Z-Test (Known Population Sigma)
Calculates z-statistic, critical z-values, and Gaussian tail probabilities.
"""

import math
from scipy import stats

def one_sample_z_test(sample_mean: float, n: int, mu_0: float, sigma: float, alpha: float = 0.05, alternative: str = "two-sided") -> dict:
    if n <= 0 or sigma <= 0:
        raise ValueError("Sample size n and sigma must be positive.")
        
    se = sigma / math.sqrt(n)
    z_stat = (sample_mean - mu_0) / se
    
    if alternative == "two-sided":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        reject = abs(z_stat) >= z_crit
    elif alternative == "greater":
        z_crit = stats.norm.ppf(1 - alpha)
        p_val = 1 - stats.norm.cdf(z_stat)
        reject = z_stat >= z_crit
    elif alternative == "less":
        z_crit = stats.norm.ppf(alpha)
        p_val = stats.norm.cdf(z_stat)
        reject = z_stat <= z_crit
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
        
    return {
        "z_statistic": round(z_stat, 4),
        "p_value": float(p_val),
        "critical_value": round(z_crit, 4),
        "standard_error": round(se, 4),
        "reject_null": bool(reject)
    }

if __name__ == "__main__":
    # Test machine part weight: known sigma = 2.5g, test mu_0 = 100g, n = 64, x_bar = 100.8g
    res = one_sample_z_test(sample_mean=100.8, n=64, mu_0=100.0, sigma=2.5, alpha=0.05, alternative="greater")
    print("=== Task 2: One-Sample Z-Test Results ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
