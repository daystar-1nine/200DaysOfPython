"""
Challenge 4: False Positive Rate Simulation under True Null Hypothesis
Runs 5,000 simulated A/B tests drawn from identical distributions (p_A = p_B = 0.05).
Confirms empirical false positive rate (Type I error) matches nominal alpha = 0.05.
"""

import math
import numpy as np
from scipy import stats

def simulate_false_positives(
    num_simulations: int = 5000,
    n_per_group: int = 1000,
    base_rate: float = 0.05,
    alpha: float = 0.05,
    seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    rejections = 0
    
    # Pre-generate binomially distributed conversion counts for both groups
    conv_a = rng.binomial(n=n_per_group, p=base_rate, size=num_simulations)
    conv_b = rng.binomial(n=n_per_group, p=base_rate, size=num_simulations)
    
    for i in range(num_simulations):
        x_a = conv_a[i]
        x_b = conv_b[i]
        p_pool = (x_a + x_b) / (2 * n_per_group)
        if p_pool == 0.0 or p_pool == 1.0:
            continue
        se = math.sqrt(p_pool * (1.0 - p_pool) * (2.0 / n_per_group))
        z = (x_b - x_a) / (n_per_group * se)  # (p_b - p_a) / se
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        if p_val < alpha:
            rejections += 1
            
    empirical_fpr = rejections / num_simulations
    error_delta = abs(empirical_fpr - alpha)
    
    res = {
        "num_simulations": num_simulations,
        "n_per_group": n_per_group,
        "baseline_rate": base_rate,
        "nominal_alpha": alpha,
        "false_positive_count": rejections,
        "empirical_false_positive_rate": round(empirical_fpr, 4),
        "absolute_error": round(error_delta, 4),
        "valid": bool(error_delta < 0.01)
    }
    
    print("=== Challenge 4: False Positive Simulation Results ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res

if __name__ == "__main__":
    simulate_false_positives()
