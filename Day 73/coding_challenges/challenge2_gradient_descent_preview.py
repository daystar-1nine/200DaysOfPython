"""
Day 73 - Challenge 2: Batch Gradient Descent Preview
Implements iterative parameter optimization for linear regression and verifies convergence to OLS.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def batch_gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1500):
    n = len(X)
    # Standardize X for smooth gradient descent surface
    x_mean = np.mean(X)
    x_std = np.std(X)
    X_norm = (X - x_mean) / x_std
    
    # Initialize parameters
    w = 0.0  # slope on normalized scale
    b = float(np.mean(y))  # initialize intercept at mean(y)
    
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = b + w * X_norm
        error = y_pred - y
        loss = np.mean(error ** 2) / 2.0
        loss_history.append(loss)
        
        # Gradients
        grad_b = np.mean(error)
        grad_w = np.mean(error * X_norm)
        
        # Updates
        b -= lr * grad_b
        w -= lr * grad_w
        
    # Convert normalized slope and intercept back to raw scale
    # y = b + w * ((X - mean) / std) = (b - w * mean / std) + (w / std) * X
    final_slope = float(w / x_std)
    final_intercept = float(b - w * x_mean / x_std)
    
    return final_slope, final_intercept, loss_history

def main():
    np.random.seed(42)
    n = 200
    X = np.random.uniform(10.0, 100.0, size=n)
    y = 50.0 + 1.75 * X + np.random.normal(0, 10.0, size=n)
    
    # 1. Analytical OLS via Scikit-Learn
    ols = LinearRegression()
    ols.fit(X.reshape(-1, 1), y)
    ols_slope = float(ols.coef_[0])
    ols_intercept = float(ols.intercept_)
    
    # 2. Gradient Descent Optimization
    gd_slope, gd_intercept, losses = batch_gradient_descent(X, y, lr=0.05, epochs=2000)
    
    print("=" * 65)
    print("DAY 73 - CHALLENGE 2: BATCH GRADIENT DESCENT PREVIEW")
    print("=" * 65)
    print(f"Training Epochs: 2,000 | Initial Loss: {losses[0]:.2f} | Final Loss: {losses[-1]:.2f}")
    print("-" * 65)
    print(f"{'Method':<25} | {'Slope (b1)':<18} | {'Intercept (b0)':<18}")
    print("-" * 65)
    print(f"{'Analytical OLS':<25} | {ols_slope:<18.4f} | {ols_intercept:<18.4f}")
    print(f"{'Gradient Descent (Iterative)':<25} | {gd_slope:<18.4f} | {gd_intercept:<18.4f}")
    print("-" * 65)
    print(f"Absolute Slope Difference:     {abs(ols_slope - gd_slope):.6f}")
    print(f"Absolute Intercept Difference: {abs(ols_intercept - gd_intercept):.6f}")
    
    assert np.isclose(gd_slope, ols_slope, atol=1e-2), "Gradient descent slope did not converge to OLS!"
    assert np.isclose(gd_intercept, ols_intercept, atol=1e-1), "Gradient descent intercept did not converge!"
    print("\nVerification: PASSED (Iterative optimization converged directly to OLS solution)")

if __name__ == "__main__":
    main()
