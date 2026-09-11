"""
Task 2: Paired Two-Sample t-Test
Simulates Before and After metrics for 30 employees, calculates individual differences,
mean difference, paired t-statistic, and p-value using scipy.stats.ttest_rel.
"""

import numpy as np
from scipy import stats

def run_paired_ttest():
    np.random.seed(42)
    n = 30
    before = np.random.normal(loc=65.0, scale=8.0, size=n)
    # After training: average gain of ~4.5 points with some variance
    improvement = np.random.normal(loc=4.5, scale=2.5, size=n)
    after = before + improvement
    
    # 1. Individual differences
    differences = after - before
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    se_diff = std_diff / np.sqrt(n)
    
    # 2. Manual paired t-stat
    t_manual = mean_diff / se_diff
    
    # 3. SciPy ttest_rel
    t_stat, p_val = stats.ttest_rel(after, before)
    
    print("=== Task 2: Paired Two-Sample t-Test ===")
    print(f"  Employees: n={n}")
    print(f"  Before Training Mean: {before.mean():.4f}")
    print(f"  After Training Mean : {after.mean():.4f}")
    print(f"  Mean Difference (d_bar): {mean_diff:.4f}")
    print(f"  Std of Differences (s_d): {std_diff:.4f}")
    print(f"  Standard Error (SE_d): {se_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f} (Manual: {t_manual:.4f})")
    print(f"  p-value: {p_val:.6e}")
    print(f"  Conclusion: {'Statistically Significant Performance Improvement' if p_val < 0.05 else 'No Significant Change'}")
    return t_stat, p_val

if __name__ == "__main__":
    run_paired_ttest()
