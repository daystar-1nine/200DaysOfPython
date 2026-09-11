"""
Task 4: Confidence Interval for Difference in Means
Computes the 95% confidence interval for (mu_B - mu_A) using Welch-Satterthwaite degrees of freedom.
"""

import math
import numpy as np
from scipy import stats

def compute_mean_difference_ci(group_a: np.ndarray, group_b: np.ndarray, confidence: float = 0.95):
    n_a = len(group_a)
    n_b = len(group_b)
    x_a = np.mean(group_a)
    x_b = np.mean(group_b)
    s_a = np.std(group_a, ddof=1)
    s_b = np.std(group_b, ddof=1)
    
    diff = x_b - x_a
    se_diff = math.sqrt((s_a**2 / n_a) + (s_b**2 / n_b))
    
    # Welch-Satterthwaite degrees of freedom
    v_a = (s_a**2) / n_a
    v_b = (s_b**2) / n_b
    df = ((v_a + v_b)**2) / ((v_a**2) / (n_a - 1) + (v_b**2) / (n_b - 1))
    
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
    
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff
    
    print("=== Task 4: Confidence Interval for Difference in Means ===")
    print(f"  Group A Mean: {x_a:.4f}, Group B Mean: {x_b:.4f}")
    print(f"  Point Estimate (B - A): {diff:.4f}")
    print(f"  Standard Error: {se_diff:.4f}")
    print(f"  Welch Degrees of Freedom: {df:.2f}")
    print(f"  Critical t ({confidence*100:.0f}%): {t_crit:.4f}")
    print(f"  {confidence*100:.0f}% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Excludes Zero: {ci_lower > 0 or ci_upper < 0}")
    return diff, ci_lower, ci_upper

if __name__ == "__main__":
    np.random.seed(42)
    g_a = np.random.normal(50.0, 8.0, 50)
    g_b = np.random.normal(55.0, 9.0, 55)
    compute_mean_difference_ci(g_a, g_b, confidence=0.95)
