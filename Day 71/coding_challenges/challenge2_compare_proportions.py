"""
Challenge 2: Two-Proportion A/B Testing Function
Builds compare_proportions(success_a, total_a, success_b, total_b, alpha=0.05) returning:
control_rate, treatment_rate, absolute_lift, relative_lift, z_statistic, p_value,
confidence interval for difference, and decision.
"""

import math
from scipy import stats

def compare_proportions(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    alpha: float = 0.05
) -> dict:
    if total_a <= 0 or total_b <= 0:
        raise ValueError("Totals must be strictly positive integers.")
    if not (0 <= success_a <= total_a) or not (0 <= success_b <= total_b):
        raise ValueError("Successes must be non-negative and not exceed totals.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")

    p_a = success_a / total_a
    p_b = success_b / total_b
    abs_lift = p_b - p_a
    rel_lift = (p_b - p_a) / p_a * 100.0 if p_a > 0 else float("inf")
    
    # Pooled proportion under H0
    p_pool = (success_a + success_b) / (total_a + total_b)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / total_a + 1.0 / total_b))
    
    z_stat = abs_lift / se_pool if se_pool > 0 else 0.0
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    
    # Unpooled CI for difference
    se_unpool = math.sqrt((p_a * (1.0 - p_a) / total_a) + (p_b * (1.0 - p_b) / total_b))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_low = abs_lift - z_crit * se_unpool
    ci_high = abs_lift + z_crit * se_unpool
    
    reject = p_val < alpha
    
    return {
        "control_rate": round(p_a, 4),
        "treatment_rate": round(p_b, 4),
        "absolute_lift": round(abs_lift, 4),
        "relative_lift": round(rel_lift, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": p_val,
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "reject_null": reject,
        "decision": "Reject H0 (Statistically Significant Lift)" if reject else "Fail to Reject H0 (Inconclusive)"
    }

if __name__ == "__main__":
    res = compare_proportions(520, 10000, 580, 10000, alpha=0.05)
    print("=== Challenge 2: compare_proportions Output ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
