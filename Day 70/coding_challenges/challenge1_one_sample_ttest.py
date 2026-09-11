"""
Challenge 1: Comprehensive One-Sample t-Test Engine
Calculates t-statistic, exact p-value, 95% confidence interval, Cohen's d effect size, and decision.
"""

from typing import Union
import numpy as np
from scipy import stats

def one_sample_ttest_engine(
    data: Union[list[float], np.ndarray],
    mu_0: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        raise ValueError("Sample size must be at least 2.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be strictly between 0 and 1.")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("Alternative must be one of: 'two-sided', 'greater', 'less'")
        
    x_bar = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    if s == 0.0:
        raise ValueError("Sample standard deviation is zero; cannot compute t-statistic.")
        
    se = s / np.sqrt(n)
    df = n - 1
    t_stat = (x_bar - mu_0) / se
    
    # P-value and Critical Value
    if alternative == "two-sided":
        p_val = float(2 * (1 - stats.t.cdf(np.abs(t_stat), df=df)))
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
        reject = abs(t_stat) >= t_crit
    elif alternative == "greater":
        p_val = float(1 - stats.t.cdf(t_stat, df=df))
        t_crit = float(stats.t.ppf(1 - alpha, df=df))
        reject = t_stat >= t_crit
    else:  # less
        p_val = float(stats.t.cdf(t_stat, df=df))
        t_crit = float(stats.t.ppf(alpha, df=df))
        reject = t_stat <= t_crit

    # 1 - alpha Confidence Interval
    t_ci_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    ci_lower = x_bar - t_ci_crit * se
    ci_upper = x_bar + t_ci_crit * se
    
    # Cohen's d
    cohen_d = (x_bar - mu_0) / s
    abs_d = abs(cohen_d)
    if abs_d < 0.20:
        effect_tier = "Negligible"
    elif abs_d < 0.50:
        effect_tier = "Small"
    elif abs_d < 0.80:
        effect_tier = "Medium"
    else:
        effect_tier = "Large"
        
    return {
        "n": n,
        "sample_mean": round(x_bar, 4),
        "sample_std": round(s, 4),
        "standard_error": round(se, 4),
        "hypothesized_mean": mu_0,
        "degrees_of_freedom": df,
        "t_statistic": round(t_stat, 4),
        "critical_value": round(t_crit, 4),
        "p_value": p_val,
        "alpha": alpha,
        "alternative": alternative,
        "reject_null": reject,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "cohens_d": round(cohen_d, 4),
        "effect_magnitude": effect_tier,
        "decision": "Reject Null Hypothesis" if reject else "Fail to Reject Null Hypothesis"
    }

if __name__ == "__main__":
    data = [28.5, 31.2, 33.8, 30.1, 29.4, 32.7, 34.1, 31.0, 30.9, 32.2]
    res = one_sample_ttest_engine(data, mu_0=30.0, alpha=0.05, alternative="greater")
    print("=== Challenge 1: One-Sample t-Test Engine ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
