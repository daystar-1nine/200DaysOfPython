"""
Task 5: One-Sample Proportion Z-Test
Verifies success-failure condition and calculates test statistic and decision.
"""

import math
from scipy import stats

def one_sample_proportion_test(successes: int, n: int, p_0: float, alpha: float = 0.05, alternative: str = "greater") -> dict:
    p_hat = successes / n
    expected_success = n * p_0
    expected_failure = n * (1 - p_0)
    assumption_met = (expected_success >= 10 and expected_failure >= 10)
    
    # Standard error under H0
    se = math.sqrt((p_0 * (1 - p_0)) / n)
    z_stat = (p_hat - p_0) / se
    
    if alternative == "greater":
        p_val = 1 - stats.norm.cdf(z_stat)
        z_crit = stats.norm.ppf(1 - alpha)
        reject = z_stat >= z_crit
    elif alternative == "less":
        p_val = stats.norm.cdf(z_stat)
        z_crit = stats.norm.ppf(alpha)
        reject = z_stat <= z_crit
    elif alternative == "two-sided":
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        reject = abs(z_stat) >= z_crit
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
        
    return {
        "sample_proportion": round(p_hat, 4),
        "hypothesized_proportion": p_0,
        "n": n,
        "assumption_met (np0, n(1-p0) >= 10)": assumption_met,
        "z_statistic": round(z_stat, 4),
        "critical_value": round(z_crit, 4),
        "p_value": float(p_val),
        "reject_null": bool(reject)
    }

if __name__ == "__main__":
    # Test if click-through rate exceeds 5% baseline: 65 clicks out of 1000 impressions
    res = one_sample_proportion_test(successes=65, n=1000, p_0=0.05, alpha=0.05, alternative="greater")
    print("=== Task 5: Proportion Test Results ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
