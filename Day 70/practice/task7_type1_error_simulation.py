"""
Task 7: Type I Error Simulation
Simulates 10,000 hypothesis tests under a true null hypothesis to confirm empirical Type I error rate matches alpha.
"""

import numpy as np
from scipy import stats

def simulate_type1_error(mu_true: float = 50.0, sigma: float = 10.0, n: int = 30, alpha: float = 0.05, num_simulations: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rejections = 0
    
    for _ in range(num_simulations):
        # Draw sample under true H0: mu = mu_true
        sample = rng.normal(loc=mu_true, scale=sigma, size=n)
        t_stat, p_val = stats.ttest_1samp(sample, popmean=mu_true)
        if p_val <= alpha:
            rejections += 1
            
    empirical_alpha = rejections / num_simulations
    error_diff = abs(empirical_alpha - alpha)
    
    return {
        "nominal_alpha": alpha,
        "num_simulations": num_simulations,
        "sample_size": n,
        "false_positive_rejections": rejections,
        "empirical_alpha": round(empirical_alpha, 4),
        "absolute_difference": round(error_diff, 4),
        "converged": bool(error_diff < 0.01)
    }

if __name__ == "__main__":
    res = simulate_type1_error(alpha=0.05, num_simulations=5000)
    print("=== Task 7: Type I Error Monte Carlo Simulation ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
