"""
Task 1: Independent Two-Sample t-Test (Welch's t-Test)
Generates Group A (mean ~ 50) and Group B (mean ~ 55), computes Welch's t-test,
and determines whether the difference is statistically significant.
"""

import numpy as np
from scipy import stats

def run_independent_ttest():
    np.random.seed(42)
    group_a = np.random.normal(loc=50.0, scale=8.0, size=50)
    group_b = np.random.normal(loc=55.0, scale=9.0, size=55)
    
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    alpha = 0.05
    is_significant = p_val < alpha
    
    print("=== Task 1: Independent Two-Sample t-Test (Welch's) ===")
    print(f"  Group A: n={len(group_a)}, Mean={group_a.mean():.4f}, Std={group_a.std(ddof=1):.4f}")
    print(f"  Group B: n={len(group_b)}, Mean={group_b.mean():.4f}, Std={group_b.std(ddof=1):.4f}")
    print(f"  Observed Difference (B - A): {group_b.mean() - group_a.mean():.4f}")
    print(f"  Welch's t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.6e}")
    print(f"  Significance (alpha={alpha}): {'Statistically Significant (Reject H0)' if is_significant else 'Not Significant'}")
    return t_stat, p_val

if __name__ == "__main__":
    run_independent_ttest()
