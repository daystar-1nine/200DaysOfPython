"""
Challenge 2: Robust One-Sample Z-Test Engine
Implements known-variance Gaussian hypothesis testing across all alternative tail specifications.
"""

import math
from typing import Union
import numpy as np
from scipy import stats

def one_sample_ztest_engine(
    data: Union[list[float], np.ndarray, float],
    mu_0: float,
    sigma: float,
    n: int = None,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    if sigma <= 0:
        raise ValueError("Known population standard deviation sigma must be strictly positive.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")
        
    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        n_sample = len(arr)
        x_bar = float(np.mean(arr))
    else:
        if n is None or n <= 0:
            raise ValueError("When data is provided as sample mean float, n must be a positive integer.")
        x_bar = float(data)
        n_sample = n
        
    se = sigma / math.sqrt(n_sample)
    z_stat = (x_bar - mu_0) / se
    
    if alternative == "two-sided":
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        reject = abs(z_stat) >= z_crit
    elif alternative == "greater":
        z_crit = float(stats.norm.ppf(1 - alpha))
        p_val = float(1 - stats.norm.cdf(z_stat))
        reject = z_stat >= z_crit
    else:
        z_crit = float(stats.norm.ppf(alpha))
        p_val = float(stats.norm.cdf(z_stat))
        reject = z_stat <= z_crit
        
    z_ci_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = x_bar - z_ci_crit * se
    ci_upper = x_bar + z_ci_crit * se
    
    return {
        "n": n_sample,
        "sample_mean": round(x_bar, 4),
        "known_sigma": sigma,
        "hypothesized_mean": mu_0,
        "standard_error": round(se, 4),
        "z_statistic": round(z_stat, 4),
        "critical_value": round(z_crit, 4),
        "p_value": p_val,
        "alpha": alpha,
        "alternative": alternative,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "reject_null": reject,
        "decision": "Reject Null Hypothesis" if reject else "Fail to Reject Null Hypothesis"
    }

if __name__ == "__main__":
    res = one_sample_ztest_engine(data=102.4, mu_0=100.0, sigma=8.0, n=64, alpha=0.05, alternative="two-sided")
    print("=== Challenge 2: One-Sample Z-Test Engine ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
