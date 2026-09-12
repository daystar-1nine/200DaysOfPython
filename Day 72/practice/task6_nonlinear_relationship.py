"""
Day 72 — Task 6: Non-Linear Relationship Analysis
Demonstrates how Pearson r collapses on symmetric quadratic functions while non-linear monotonicity holds.
"""
import numpy as np
from scipy import stats

def main():
    np.random.seed(42)
    x = np.linspace(-10, 10, 201)
    y = x ** 2
    
    r_sym, p_sym = stats.pearsonr(x, y)
    rho_sym, rho_p_sym = stats.spearmanr(x, y)
    
    # Now positive monotonic half
    x_pos = np.linspace(0, 10, 101)
    y_pos = x_pos ** 2
    r_pos, p_pos = stats.pearsonr(x_pos, y_pos)
    rho_pos, rho_p_pos = stats.spearmanr(x_pos, y_pos)
    
    print("=" * 60)
    print("DAY 72 — TASK 6: NON-LINEAR RELATIONSHIP (Y = X^2)")
    print("=" * 60)
    print("Symmetric Domain [-10, 10]:")
    print(f"  Pearson r:    {r_sym:.4f} (p={p_sym:.4f})  -> Close to 0!")
    print(f"  Spearman rho: {rho_sym:.4f} (p={rho_p_sym:.4f}) -> Close to 0 (non-monotonic)!")
    print("Positive Domain [0, 10]:")
    print(f"  Pearson r:    {r_pos:.4f} (Strong linear approximation)")
    print(f"  Spearman rho: {rho_pos:.4f} (Perfect monotonic rank correlation!)")
    
    assert abs(r_sym) < 0.05, "Symmetric parabola should have near zero linear correlation!"
    assert np.isclose(rho_pos, 1.0), "Monotonic quadratic function should have Spearman rho = 1.0!"
    print("Conclusion: r ~= 0 does NOT imply independence. Visual inspection is mandatory.")

if __name__ == "__main__":
    main()
