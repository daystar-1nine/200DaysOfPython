"""
Day 73 - Task 4: Calculate Metrics Manually
Implements MAE, MSE, RMSE, and R2 from scratch and compares with scikit-learn.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(calculate_mse(y_true, y_pred)))

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def main():
    y_true = np.array([100.0, 120.0, 140.0, 160.0, 180.0])
    y_pred = np.array([95.0, 125.0, 138.0, 162.0, 175.0])
    
    # Custom calculations
    mae_custom = calculate_mae(y_true, y_pred)
    mse_custom = calculate_mse(y_true, y_pred)
    rmse_custom = calculate_rmse(y_true, y_pred)
    r2_custom = calculate_r2(y_true, y_pred)
    
    # Scikit-learn calculations
    mae_sk = mean_absolute_error(y_true, y_pred)
    mse_sk = mean_squared_error(y_true, y_pred)
    rmse_sk = np.sqrt(mse_sk)
    r2_sk = r2_score(y_true, y_pred)
    
    print("=" * 60)
    print("DAY 73 - TASK 4: REGRESSION EVALUATION METRICS")
    print("=" * 60)
    print(f"{'Metric':<8} | {'Custom Implementation':<24} | {'Scikit-Learn':<15}")
    print("-" * 60)
    print(f"{'MAE':<8} | {mae_custom:<24.4f} | {mae_sk:<15.4f}")
    print(f"{'MSE':<8} | {mse_custom:<24.4f} | {mse_sk:<15.4f}")
    print(f"{'RMSE':<8} | {rmse_custom:<24.4f} | {rmse_sk:<15.4f}")
    print(f"{'R2':<8} | {r2_custom:<24.4f} | {r2_sk:<15.4f}")
    
    assert np.isclose(mae_custom, mae_sk), "MAE mismatch!"
    assert np.isclose(mse_custom, mse_sk), "MSE mismatch!"
    assert np.isclose(rmse_custom, rmse_sk), "RMSE mismatch!"
    assert np.isclose(r2_custom, r2_sk), "R2 mismatch!"
    print("\nVerification: PASSED (Custom metrics match scikit-learn to precision)")

if __name__ == "__main__":
    main()
