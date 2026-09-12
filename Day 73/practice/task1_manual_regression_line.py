"""
Day 73 - Task 1: Calculate a Regression Line Manually
Calculates OLS slope and intercept step-by-step from first principles and verifies with scikit-learn.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def manual_linear_regression(x: np.ndarray, y: np.ndarray):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)
    
    slope = float(numerator / denominator)
    intercept = float(mean_y - slope * mean_x)
    
    return slope, intercept, mean_x, mean_y

def main():
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 5, 8, 10], dtype=float)
    
    slope_m, intercept_m, mx, my = manual_linear_regression(X, y)
    
    # Scikit-Learn verification
    sk_model = LinearRegression()
    sk_model.fit(X.reshape(-1, 1), y)
    slope_sk = float(sk_model.coef_[0])
    intercept_sk = float(sk_model.intercept_)
    
    print("=" * 60)
    print("DAY 73 - TASK 1: MANUAL REGRESSION LINE DERIVATION")
    print("=" * 60)
    print(f"X: {X}")
    print(f"y: {y}")
    print(f"Mean(X): {mx:.2f}, Mean(Y): {my:.2f}")
    print(f"Manual Slope (b1):     {slope_m:.4f}")
    print(f"Manual Intercept (b0): {intercept_m:.4f}")
    print(f"Regression Equation:   y_hat = {intercept_m:.4f} + {slope_m:.4f} * X")
    print("-" * 60)
    print(f"Scikit-Learn Slope:    {slope_sk:.4f}")
    print(f"Scikit-Learn Intercept:{intercept_sk:.4f}")
    
    assert np.isclose(slope_m, slope_sk), "Slope calculation mismatch!"
    assert np.isclose(intercept_m, intercept_sk), "Intercept calculation mismatch!"
    print("Verification: PASSED (Exact alignment between analytical math and scikit-learn)")

if __name__ == "__main__":
    main()
