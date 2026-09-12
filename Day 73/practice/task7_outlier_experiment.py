"""
Day 73 - Task 7: Outlier Impact Experiment
Demonstrates how a single high-leverage outlier warps regression slope, inflates RMSE, and crashes R2.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    np.random.seed(42)
    n = 50
    X = np.linspace(10, 100, n).reshape(-1, 1)
    y = (30 + 1.5 * X.ravel() + np.random.normal(0, 5, n))
    
    # Clean fit
    model_clean = LinearRegression()
    model_clean.fit(X, y)
    y_pred_clean = model_clean.predict(X)
    r2_clean = r2_score(y, y_pred_clean)
    rmse_clean = np.sqrt(mean_squared_error(y, y_pred_clean))
    slope_clean = model_clean.coef_[0]
    
    # Contaminated with 1 high-leverage discordant outlier
    X_corrupt = np.vstack([X, [[150.0]]])
    y_corrupt = np.append(y, 10.0) # Discordant extreme point
    
    model_corrupt = LinearRegression()
    model_corrupt.fit(X_corrupt, y_corrupt)
    y_pred_corrupt = model_corrupt.predict(X_corrupt)
    r2_corrupt = r2_score(y_corrupt, y_pred_corrupt)
    rmse_corrupt = np.sqrt(mean_squared_error(y_corrupt, y_pred_corrupt))
    slope_corrupt = model_corrupt.coef_[0]
    
    print("=" * 65)
    print("DAY 73 - TASK 7: HIGH-LEVERAGE OUTLIER SENSITIVITY EXPERIMENT")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Clean Data (N=50)':<20} | {'Contaminated (1 Outlier)':<24}")
    print("-" * 65)
    print(f"{'Slope (b1)':<18} | {slope_clean:<20.4f} | {slope_corrupt:<24.4f}")
    print(f"{'RMSE':<18} | {rmse_clean:<20.4f} | {rmse_corrupt:<24.4f}")
    print(f"{'R2 Score':<18} | {r2_clean:<20.4f} | {r2_corrupt:<24.4f}")
    print("-" * 65)
    print("Impact of 1 Single Outlier at (150, 10):")
    print(f"  - Slope rotated downwards by {abs(slope_clean - slope_corrupt):.4f}")
    print(f"  - RMSE inflated by {rmse_corrupt / rmse_clean:.2f}x")
    print(f"  - R2 crashed from {r2_clean:.3f} down to {r2_corrupt:.3f}")
    
    assert r2_clean > 0.95, "Clean dataset should have R2 > 0.95!"
    assert r2_corrupt < 0.85, "Outlier should degrade R2 significantly!"
    assert rmse_corrupt > rmse_clean * 1.5, "Outlier should inflate RMSE!"
    print("Verification: PASSED (Demonstrated high vulnerability of OLS to leverage points)")

if __name__ == "__main__":
    main()
