"""
Challenge 5: Multi-Alpha Type I Error Monte Carlo Simulation
Tests empirical false positive rates across alpha in {0.01, 0.05, 0.10} with 5000 iterations each.
"""

import numpy as np
from scipy import stats

def multi_alpha_simulation(
    alphas: list[float] = (0.01, 0.05, 0.10),
    num_simulations: int = 5000,
    sample_size: int = 30,
    seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    
    # Pre-generate standard normal samples under true H0: mu = 0.0
    all_samples = rng.normal(loc=0.0, scale=1.0, size=(num_simulations, sample_size))
    
    # Calculate all t-stats and p-values
    means = np.mean(all_samples, axis=1)
    stds = np.std(all_samples, axis=1, ddof=1)
    se = stds / np.sqrt(sample_size)
    t_stats = means / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=sample_size - 1))
    
    for a in alphas:
        rejections = np.sum(p_values <= a)
        emp_rate = float(rejections / num_simulations)
        abs_diff = abs(emp_rate - a)
        results[a] = {
            "nominal_alpha": a,
            "simulations": num_simulations,
            "false_positives": int(rejections),
            "empirical_rate": round(emp_rate, 4),
            "error_delta": round(abs_diff, 4),
            "valid": bool(abs_diff < 0.01)
        }
        
    return results

if __name__ == "__main__":
    sim_res = multi_alpha_simulation(alphas=[0.01, 0.05, 0.10], num_simulations=5000)
    print("=== Challenge 5: Multi-Alpha Type I Error Simulation ===")
    for alpha, res in sim_res.items():
        print(f"  Alpha {alpha:4.2f} -> Rejection Rate: {res['empirical_rate']:.4f} | Error Delta: {res['error_delta']:.4f} | Valid: {res['valid']}")
