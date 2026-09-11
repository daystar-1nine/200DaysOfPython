"""
Task 5: A/B Conversion Experiment Analysis
Analyzes conversion rates for Control (n=10k, conv=520) vs Treatment (n=10k, conv=580),
calculating absolute lift, relative lift, pooled z-test, and 95% CI for difference.
"""

import math
from scipy import stats

def analyze_ab_conversion(n_a: int = 10000, conv_a: int = 520, n_b: int = 10000, conv_b: int = 580):
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    
    abs_lift = p_b - p_a
    rel_lift = (p_b - p_a) / p_a * 100.0
    
    # 1. Pooled Two-Proportion Z-Test under H0: p_A = p_B
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z_stat = abs_lift / se_pool
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    
    # 2. 95% CI for Difference (Unpooled)
    se_unpool = math.sqrt((p_a * (1.0 - p_a) / n_a) + (p_b * (1.0 - p_b) / n_b))
    z_crit = stats.norm.ppf(0.975)
    ci_lower = abs_lift - z_crit * se_unpool
    ci_upper = abs_lift + z_crit * se_unpool
    
    print("=== Task 5: A/B Conversion Experiment Analysis ===")
    print(f"  Control (A)   : {conv_a:,} / {n_a:,} ({p_a * 100:.2f}%)")
    print(f"  Treatment (B) : {conv_b:,} / {n_b:,} ({p_b * 100:.2f}%)")
    print(f"  Absolute Lift : {abs_lift * 100:+.2f} percentage points ({abs_lift:+.4f})")
    print(f"  Relative Lift : {rel_lift:+.2f}%")
    print(f"  Pooled SE     : {se_pool:.6f}")
    print(f"  z-statistic   : {z_stat:.4f}")
    print(f"  p-value       : {p_val:.6f}")
    print(f"  95% CI (Diff) : [{ci_lower * 100:+.2f}%, {ci_upper * 100:+.2f}%]")
    print(f"  Decision (alpha=0.05): {'Reject H0 (Statistically Significant Lift)' if p_val < 0.05 else 'Fail to Reject H0'}")

if __name__ == "__main__":
    analyze_ab_conversion()
