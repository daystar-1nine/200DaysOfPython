"""
Day 72 — Task 2: Pearson Correlation Coefficient
Computes Pearson r manually from normalized covariance and verifies with NumPy and SciPy.
"""
import numpy as np
from scipy import stats

def manual_pearson_r(x: np.ndarray, y: np.ndarray):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    num = np.sum((x - mean_x) * (y - mean_y))
    denom = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    if denom == 0:
        return 0.0, 1.0, 0.0
    
    r = float(num / denom)
    
    # Test statistic under H0: rho = 0
    if abs(r) >= 1.0:
        t_stat = np.inf if r > 0 else -np.inf
        p_val = 0.0
    else:
        df = n - 2
        t_stat = r * np.sqrt(df / (1.0 - r ** 2))
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
        
    return r, p_val, t_stat

def main():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    
    r_manual, p_manual, t_manual = manual_pearson_r(x, y)
    r_numpy = np.corrcoef(x, y)[0, 1]
    res_scipy = stats.pearsonr(x, y)
    
    print("=" * 55)
    print("DAY 72 — TASK 2: PEARSON CORRELATION")
    print("=" * 55)
    print(f"Manual Pearson r:   {r_manual:.6f} (t={t_manual:.4f}, p={p_manual:.4e})")
    print(f"NumPy Pearson r:    {r_numpy:.6f}")
    print(f"SciPy Pearson r:    {res_scipy.statistic:.6f} (p={res_scipy.pvalue:.4e})")
    
    assert np.isclose(r_manual, r_numpy), "Manual and NumPy r mismatch!"
    assert np.isclose(r_manual, res_scipy.statistic), "Manual and SciPy r mismatch!"
    print("Verification: PASSED (All statistical engines align perfectly)")

if __name__ == "__main__":
    main()
