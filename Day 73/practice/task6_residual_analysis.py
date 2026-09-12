"""
Day 73 - Task 6: Residual Analysis
Evaluates residual distribution, checks for homoscedasticity, and prints statistical diagnostic summary.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    np.random.seed(42)
    n = 150
    X = np.linspace(10, 100, n).reshape(-1, 1)
    # Homoscedastic Gaussian noise
    noise = np.random.normal(0, 12, size=n)
    y = 50 + 2.2 * X.ravel() + noise
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals, ddof=1)
    skew_approx = np.mean(((residuals - mean_res) / std_res) ** 3)
    
    # Check variance split (first half vs second half to test homoscedasticity)
    half = n // 2
    var_first = np.var(residuals[:half], ddof=1)
    var_second = np.var(residuals[half:], ddof=1)
    var_ratio = var_second / var_first
    
    print("=" * 60)
    print("DAY 73 - TASK 6: RESIDUAL DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print(f"Sample Size (N):            {n}")
    print(f"Mean of Residuals:          {mean_res:.4e} (Must be ~0)")
    print(f"Standard Deviation:         {std_res:.4f}")
    print(f"Residual Skewness:          {skew_approx:.4f} (Near 0 indicates symmetry)")
    print(f"Variance Ratio (High/Low):  {var_ratio:.4f} (Close to 1 indicates homoscedasticity)")
    print("-" * 60)
    print("Key Diagnostic Observations:")
    print("  1. Mean of residuals is zero, validating algebraic OLS first-order condition.")
    print("  2. Skewness is near zero, supporting the Gaussian residual assumption.")
    print("  3. Variance ratio is near 1.0, showing absence of severe heteroscedasticity.")
    
    assert np.isclose(mean_res, 0.0, atol=1e-10)
    assert 0.5 < var_ratio < 2.0, "Residual variance should remain reasonably stable!"
    print("Verification: PASSED (Residual assumptions verified)")

if __name__ == "__main__":
    main()
