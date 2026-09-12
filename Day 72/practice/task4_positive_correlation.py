"""
Day 72 — Task 4: Positive Correlation Simulation
Generates synthetic data with strong positive linear association and computes 95% Fisher CI.
"""
import numpy as np
from scipy import stats

def fisher_ci(r: float, n: int, confidence: float = 0.95):
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    alpha = 1.0 - confidence
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    z_low = z - z_crit * se
    z_high = z + z_crit * se
    return float(np.tanh(z_low)), float(np.tanh(z_high))

def main():
    np.random.seed(42)
    n = 100
    x = np.arange(1, n + 1, dtype=float)
    noise = np.random.normal(0, 15, size=n)
    y = 2.0 * x + noise
    
    r, p_val = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    ci_low, ci_high = fisher_ci(r, n)
    
    print("=" * 55)
    print("DAY 72 — TASK 4: POSITIVE CORRELATION SIMULATION")
    print("=" * 55)
    print(f"Sample size (N):     {n}")
    print(f"Pearson r:           {r:.4f} (p={p_val:.4e})")
    print(f"Spearman rho:        {rho:.4f} (p={rho_p:.4e})")
    print(f"95% Fisher CI for r: [{ci_low:.4f}, {ci_high:.4f}]")
    
    assert r > 0.85, "Expected strong positive correlation!"
    assert p_val < 0.001, "Expected statistically significant correlation!"
    print("Result: Validated Strong Positive Linear Association.")

if __name__ == "__main__":
    main()
