"""
Day 72 — Task 5: Negative Correlation Simulation
Generates synthetic data with strong negative linear association and evaluates statistical significance.
"""
import numpy as np
from scipy import stats

def main():
    np.random.seed(42)
    n = 100
    x = np.arange(1, n + 1, dtype=float)
    noise = np.random.normal(0, 15, size=n)
    y = 100.0 - 2.0 * x + noise
    
    r, p_val = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    
    print("=" * 55)
    print("DAY 72 — TASK 5: NEGATIVE CORRELATION SIMULATION")
    print("=" * 55)
    print(f"Sample size (N): {n}")
    print(f"Pearson r:       {r:.4f} (p={p_val:.4e})")
    print(f"Spearman rho:    {rho:.4f} (p={rho_p:.4e})")
    
    assert r < -0.85, "Expected strong negative correlation!"
    assert p_val < 0.001, "Expected statistically significant negative correlation!"
    print("Result: Validated Strong Negative Linear Association.")

if __name__ == "__main__":
    main()
