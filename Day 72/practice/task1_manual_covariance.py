"""
Day 72 — Task 1: Manual Covariance Calculation
Calculates sample covariance step-by-step from first principles and verifies with NumPy.
"""
import numpy as np

def manual_sample_covariance(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n != len(y):
        raise ValueError("Arrays must have identical lengths.")
    if n < 2:
        raise ValueError("Sample size must be at least 2.")
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    cross_products = [(xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)]
    cov_val = sum(cross_products) / (n - 1)
    return float(cov_val)

def main():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 5, 8, 10], dtype=float)
    
    cov_manual = manual_sample_covariance(x, y)
    cov_numpy = np.cov(x, y, ddof=1)[0, 1]
    
    print("=" * 55)
    print("DAY 72 — TASK 1: MANUAL COVARIANCE")
    print("=" * 55)
    print(f"X: {x}")
    print(f"Y: {y}")
    print(f"Mean(X): {np.mean(x):.2f}, Mean(Y): {np.mean(y):.2f}")
    print(f"Manual Sample Covariance s_xy: {cov_manual:.4f}")
    print(f"NumPy Sample Covariance np.cov: {cov_numpy:.4f}")
    
    assert np.isclose(cov_manual, cov_numpy), "Manual and NumPy covariance mismatch!"
    print("Verification: PASSED (Exact match within floating tolerance)")

if __name__ == "__main__":
    main()
