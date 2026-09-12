"""
Day 73 - Task 3: Residuals
Computes actual, predicted, and residual values for every observation, proving sum(residuals) = 0.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2, 4, 5, 8, 10], dtype=float)
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print("=" * 60)
    print("DAY 73 - TASK 3: RESIDUALS CALCULATION")
    print("=" * 60)
    print(f"{'Index':<6} | {'Actual (y)':<12} | {'Predicted (y_hat)':<18} | {'Residual (e)':<14}")
    print("-" * 60)
    for i in range(len(y)):
        print(f"{i:<6} | {y[i]:<12.2f} | {y_pred[i]:<18.2f} | {residuals[i]:<14.4f}")
    print("-" * 60)
    sum_res = float(np.sum(residuals))
    mean_res = float(np.mean(residuals))
    print(f"Sum of Residuals:  {sum_res:.6e} (Essentially 0)")
    print(f"Mean of Residuals: {mean_res:.6e} (Essentially 0)")
    
    assert np.isclose(sum_res, 0.0, atol=1e-10), "Sum of OLS residuals must be zero!"
    print("Verification: PASSED (Residual zero-sum property confirmed)")

if __name__ == "__main__":
    main()
