"""
Task 3: Compare Test Types (Independent vs Paired)
Analyzes the exact same paired dataset under two designs:
1. Treated as independent groups (ignoring within-subject pairing)
2. Treated as paired groups (taking differences)
Shows why pairing accounts for covariance and achieves superior power.
"""

import numpy as np
from scipy import stats

def compare_test_designs():
    np.random.seed(42)
    n = 30
    # Wide spread between individuals (std=15.0), but consistent small improvement (+3.0)
    baseline_ability = np.random.normal(loc=100.0, scale=15.0, size=n)
    before = baseline_ability + np.random.normal(loc=0.0, scale=2.0, size=n)
    after = before + np.random.normal(loc=3.0, scale=1.5, size=n)
    
    # Independent Test (Welch)
    t_ind, p_ind = stats.ttest_ind(after, before, equal_var=False)
    
    # Paired Test
    t_paired, p_paired = stats.ttest_rel(after, before)
    
    print("=== Task 3: Test Design Comparison (Independent vs Paired) ===")
    print(f"  Sample size per condition: n={n}")
    print(f"  Correlation between Before and After: {np.corrcoef(before, after)[0, 1]:.4f}")
    print("\n  [Design 1: Independent t-Test (Ignored Pairing)]")
    print(f"    t-statistic: {t_ind:.4f}")
    print(f"    p-value:     {p_ind:.6f}")
    print(f"    Significant at alpha=0.05: {p_ind < 0.05}")
    print("\n  [Design 2: Paired t-Test (Accounted for Pairing)]")
    print(f"    t-statistic: {t_paired:.4f}")
    print(f"    p-value:     {p_paired:.6e}")
    print(f"    Significant at alpha=0.05: {p_paired < 0.05}")
    print("\n  [Statistical Takeaway]")
    print("    Pairing eliminates between-subject variation. When subjects are correlated,")
    print("    the paired test has dramatically lower standard error and vastly higher power!")

if __name__ == "__main__":
    compare_test_designs()
