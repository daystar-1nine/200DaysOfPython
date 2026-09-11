"""
Task 7: End-to-End A/B Test Simulation
Simulates 10,000 users, performs random 50/50 allocation, generates conversions,
and computes complete metrics dictionary.
"""

import numpy as np
from scipy import stats

def simulate_full_ab_test(n_users: int = 10000, p_a: float = 0.052, p_b: float = 0.058, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    # Random 50/50 assignment
    groups = rng.choice(["Control", "Treatment"], size=n_users, p=[0.5, 0.5])
    
    mask_a = (groups == "Control")
    mask_b = (groups == "Treatment")
    n_a = int(np.sum(mask_a))
    n_b = int(np.sum(mask_b))
    
    # Bernoulli conversions
    conv_a = int(np.sum(rng.binomial(n=1, p=p_a, size=n_a)))
    conv_b = int(np.sum(rng.binomial(n=1, p=p_b, size=n_b)))
    
    rate_a = conv_a / n_a
    rate_b = conv_b / n_b
    diff = rate_b - rate_a
    rel_lift = (rate_b - rate_a) / rate_a * 100.0
    
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z_stat = diff / se_pool
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    
    se_unpool = np.sqrt((rate_a * (1.0 - rate_a) / n_a) + (rate_b * (1.0 - rate_b) / n_b))
    ci_low = diff - 1.96 * se_unpool
    ci_high = diff + 1.96 * se_unpool
    
    res = {
        "n_control": n_a,
        "n_treatment": n_b,
        "conversions_control": conv_a,
        "conversions_treatment": conv_b,
        "conversion_rate_control": round(rate_a, 4),
        "conversion_rate_treatment": round(rate_b, 4),
        "absolute_lift": round(diff, 4),
        "relative_lift_pct": round(rel_lift, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": float(p_val),
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "reject_null": bool(p_val < 0.05)
    }
    
    print("=== Task 7: Full A/B Test Simulation Results ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res

if __name__ == "__main__":
    simulate_full_ab_test()
