"""
Challenge 1: Universal Two-Sample Mean Comparison Engine
Builds compare_means(group_a, group_b, alpha=0.05, equal_var=False) returning
complete diagnostics: means, difference, standard error, degrees of freedom,
t-statistic, p-value, confidence interval, Cohen's d, and decision.
"""

import math
from typing import Sequence, Union
import numpy as np
from scipy import stats

def compare_means(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]],
    alpha: float = 0.05,
    equal_var: bool = False
) -> dict:
    arr_a = np.asarray(group_a, dtype=float)
    arr_b = np.asarray(group_b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    
    n_a = len(arr_a)
    n_b = len(arr_b)
    if n_a < 2 or n_b < 2:
        raise ValueError("Both groups must contain at least 2 valid observations.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
        
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    s_a = math.sqrt(var_a)
    s_b = math.sqrt(var_b)
    
    diff = mean_b - mean_a
    
    if equal_var:
        # Pooled Student's t-test
        df = n_a + n_b - 2
        s_p2 = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
        se_diff = math.sqrt(s_p2 * (1.0 / n_a + 1.0 / n_b))
        s_pooled = math.sqrt(s_p2)
    else:
        # Welch's t-test
        se_diff = math.sqrt((var_a / n_a) + (var_b / n_b))
        v_a = var_a / n_a
        v_b = var_b / n_b
        df = ((v_a + v_b) ** 2) / ((v_a ** 2) / (n_a - 1) + (v_b ** 2) / (n_b - 1))
        # Pooled SD for Cohen's d
        s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
    t_stat = diff / se_diff
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
    
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff
    
    # Cohen's d
    cohen_d = diff / s_pooled if s_pooled > 0 else 0.0
    abs_d = abs(cohen_d)
    if abs_d < 0.20:
        effect_tier = "Negligible"
    elif abs_d < 0.50:
        effect_tier = "Small"
    elif abs_d < 0.80:
        effect_tier = "Medium"
    else:
        effect_tier = "Large"
        
    reject = p_val < alpha
    
    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "difference": round(diff, 4),
        "standard_error": round(se_diff, 4),
        "degrees_of_freedom": round(df, 2),
        "statistic": round(t_stat, 4),
        "p_value": p_val,
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "cohens_d": round(cohen_d, 4),
        "effect_magnitude": effect_tier,
        "decision": "Reject H0 (Statistically Significant)" if reject else "Fail to Reject H0 (No Significant Difference)"
    }

if __name__ == "__main__":
    np.random.seed(42)
    a = np.random.normal(50, 8, 40)
    b = np.random.normal(56, 9, 45)
    res = compare_means(a, b, alpha=0.05, equal_var=False)
    print("=== Challenge 1: compare_means Output ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
