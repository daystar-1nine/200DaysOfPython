"""
Task 1: Manual Step-by-Step Hypothesis Test
Demonstrates the full manual arithmetic of a 1-sample t-test from first principles.
"""

import math
from scipy import stats

def manual_one_sample_t_test(data: list[float], mu_0: float, alpha: float = 0.05, alternative: str = "two-sided") -> dict:
    n = len(data)
    if n < 2:
        raise ValueError("Sample size must be at least 2.")
    
    # 1. Sample Mean
    x_bar = sum(data) / n
    
    # 2. Sample Variance & Standard Deviation
    variance = sum((x - x_bar) ** 2 for x in data) / (n - 1)
    s = math.sqrt(variance)
    
    # 3. Standard Error
    se = s / math.sqrt(n)
    
    # 4. Degrees of Freedom
    df = n - 1
    
    # 5. Test Statistic
    t_stat = (x_bar - mu_0) / se
    
    # 6. Critical Value & P-value based on alternative hypothesis
    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df=df)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        reject = abs(t_stat) >= t_crit
        crit_desc = f"+/- {t_crit:.4f}"
    elif alternative == "greater":
        t_crit = stats.t.ppf(1 - alpha, df=df)
        p_val = 1 - stats.t.cdf(t_stat, df=df)
        reject = t_stat >= t_crit
        crit_desc = f"> {t_crit:.4f}"
    elif alternative == "less":
        t_crit = stats.t.ppf(alpha, df=df)
        p_val = stats.t.cdf(t_stat, df=df)
        reject = t_stat <= t_crit
        crit_desc = f"< {t_crit:.4f}"
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
        
    return {
        "sample_size": n,
        "sample_mean": round(x_bar, 4),
        "sample_std": round(s, 4),
        "standard_error": round(se, 4),
        "degrees_of_freedom": df,
        "hypothesized_mean": mu_0,
        "t_statistic": round(t_stat, 4),
        "critical_value_threshold": crit_desc,
        "p_value": float(p_val),
        "alpha": alpha,
        "reject_null": bool(reject),
        "conclusion": "Reject H0 (Statistically Significant)" if reject else "Fail to Reject H0 (Not Statistically Significant)"
    }

if __name__ == "__main__":
    sample = [32.1, 29.8, 33.5, 31.2, 34.0, 30.5, 33.1, 32.8, 35.2, 31.9]
    res = manual_one_sample_t_test(sample, mu_0=30.0, alpha=0.05, alternative="two-sided")
    print("=== Task 1: Manual Hypothesis Test Results ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
