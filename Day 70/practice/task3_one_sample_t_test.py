"""
Task 3: One-Sample t-Test (SciPy Validation)
Compares custom t-test implementation directly with scipy.stats.ttest_1samp.
"""

import numpy as np
from scipy import stats

def compare_t_test(data: np.ndarray, mu_0: float, alpha: float = 0.05) -> dict:
    # Custom calculation
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    t_manual = (x_bar - mu_0) / se
    df = n - 1
    p_manual = 2 * (1 - stats.t.cdf(np.abs(t_manual), df=df))
    
    # SciPy calculation
    res_scipy = stats.ttest_1samp(data, popmean=mu_0)
    
    diff_t = abs(t_manual - res_scipy.statistic)
    diff_p = abs(p_manual - res_scipy.pvalue)
    
    return {
        "manual_t": round(float(t_manual), 6),
        "scipy_t": round(float(res_scipy.statistic), 6),
        "manual_p": round(float(p_manual), 6),
        "scipy_p": round(float(res_scipy.pvalue), 6),
        "t_diff": diff_t,
        "p_diff": diff_p,
        "matches": (diff_t < 1e-9 and diff_p < 1e-9)
    }

if __name__ == "__main__":
    np.random.seed(42)
    sample = np.random.normal(loc=52.5, scale=4.0, size=35)
    comp = compare_t_test(sample, mu_0=50.0)
    print("=== Task 3: t-Test SciPy Comparison ===")
    for k, v in comp.items():
        print(f"  {k}: {v}")
