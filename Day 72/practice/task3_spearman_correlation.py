"""
Day 72 — Task 3: Spearman Rank Correlation
Demonstrates rank transformation and computes Spearman rho manually and via SciPy.
"""
import numpy as np
import pandas as pd
from scipy import stats

def manual_spearman_rho(x: np.ndarray, y: np.ndarray):
    rank_x = pd.Series(x).rank(method='average').values
    rank_y = pd.Series(y).rank(method='average').values
    
    # Pearson on ranks
    num = np.sum((rank_x - np.mean(rank_x)) * (rank_y - np.mean(rank_y)))
    denom = np.sqrt(np.sum((rank_x - np.mean(rank_x)) ** 2) * np.sum((rank_y - np.mean(rank_y)) ** 2))
    rho = float(num / denom)
    
    n = len(x)
    df = n - 2
    if abs(rho) >= 1.0:
        p_val = 0.0
    else:
        t_stat = rho * np.sqrt(df / (1.0 - rho ** 2))
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
        
    return rho, p_val

def main():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([30, 35, 45, 60, 100], dtype=float)
    
    rho_manual, p_manual = manual_spearman_rho(x, y)
    res_scipy = stats.spearmanr(x, y)
    res_pearson = stats.pearsonr(x, y)
    
    print("=" * 55)
    print("DAY 72 — TASK 3: SPEARMAN RANK CORRELATION")
    print("=" * 55)
    print(f"X (Experience): {x}")
    print(f"Y (Salary):     {y}")
    print(f"Manual Spearman rho: {rho_manual:.6f}")
    print(f"SciPy Spearman rho:  {res_scipy.statistic:.6f} (p={res_scipy.pvalue:.4e})")
    print(f"SciPy Pearson r:     {res_pearson.statistic:.6f} (p={res_pearson.pvalue:.4e})")
    
    assert np.isclose(rho_manual, res_scipy.statistic), "Spearman rho mismatch!"
    print("Verification: PASSED (Spearman captures monotonic increase perfectly)")

if __name__ == "__main__":
    main()
