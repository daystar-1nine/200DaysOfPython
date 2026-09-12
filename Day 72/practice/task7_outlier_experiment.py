"""
Day 72 — Task 7: Outlier Impact Experiment
Demonstrates how a single high-leverage outlier severely corrupts Pearson r while Spearman rho remains robust.
"""
import numpy as np
from scipy import stats

def main():
    # Base clean positive dataset
    x = np.linspace(1, 20, 20)
    y = 2.0 * x + 1.0
    
    r_before = stats.pearsonr(x, y).statistic
    rho_before = stats.spearmanr(x, y).statistic
    
    # Introduce single extreme discordant leverage point
    x_corrupt = np.append(x, 150.0)
    y_corrupt = np.append(y, -150.0)
    
    r_after = stats.pearsonr(x_corrupt, y_corrupt).statistic
    rho_after = stats.spearmanr(x_corrupt, y_corrupt).statistic
    
    print("=" * 60)
    print("DAY 72 - TASK 7: OUTLIER EXPERIMENT")
    print("=" * 60)
    print("Clean Data (N=20):")
    print(f"  Pearson r:    {r_before:.4f}")
    print(f"  Spearman rho: {rho_before:.4f}")
    print("\nContaminated Data with Outlier (150, -150) (N=21):")
    print(f"  Pearson r:    {r_after:.4f}  <-- Inverted to negative!")
    print(f"  Spearman rho: {rho_after:.4f}  <-- Stays strongly positive!")
    
    assert r_before > 0.99, "Clean data should be perfectly correlated!"
    assert r_after < 0.0, "Pearson r should be flipped negative by severe outlier!"
    assert rho_after > 0.65, "Spearman should resist outlier distortion!"
    print("Conclusion: Pearson is fragile to leverage points; Spearman offers non-parametric rank resistance.")

if __name__ == "__main__":
    main()
