"""
Two-sample pooled proportion hypothesis tests for A/B experiments.
"""

import math
from scipy import stats

try:
    from app.validator import validate_alpha, validate_proportions
except (ImportError, ModuleNotFoundError):
    from validator import validate_alpha, validate_proportions

def two_proportion_ztest(
    success_c: int,
    total_c: int,
    success_t: int,
    total_t: int,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    alpha = validate_alpha(alpha)
    p_c = validate_proportions(success_c, total_c)
    p_t = validate_proportions(success_t, total_t)
    
    abs_lift = p_t - p_c
    rel_lift = (p_t - p_c) / p_c * 100.0 if p_c > 0 else float("inf")
    
    # Pooled proportion under H0
    p_pool = (success_c + success_t) / (total_c + total_t)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / total_c + 1.0 / total_t))
    
    z_stat = abs_lift / se_pool if se_pool > 0 else 0.0
    
    if alternative == "two-sided":
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        reject = abs(z_stat) >= z_crit
    elif alternative == "greater":
        p_val = float(1 - stats.norm.cdf(z_stat))
        z_crit = float(stats.norm.ppf(1 - alpha))
        reject = z_stat >= z_crit
    elif alternative == "less":
        p_val = float(stats.norm.cdf(z_stat))
        z_crit = float(stats.norm.ppf(alpha))
        reject = z_stat <= z_crit
    else:
        raise ValueError(f"Invalid alternative: '{alternative}'. Must be 'two-sided', 'greater', or 'less'.")
        
    return {
        "test_type": "Two-Proportion Pooled Z-test",
        "control_conversions": success_c,
        "control_total": total_c,
        "treatment_conversions": success_t,
        "treatment_total": total_t,
        "rate_control": round(p_c, 4),
        "rate_treatment": round(p_t, 4),
        "absolute_lift": round(abs_lift, 4),
        "relative_lift_pct": round(rel_lift, 2),
        "pooled_proportion": round(p_pool, 4),
        "pooled_standard_error": round(se_pool, 6),
        "z_statistic": round(z_stat, 4),
        "critical_value": round(z_crit, 4),
        "p_value": p_val,
        "alternative": alternative,
        "alpha": alpha,
        "reject_null": reject
    }
