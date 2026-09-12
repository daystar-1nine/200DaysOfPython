import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

def calculate_mse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred) -> float:
    return r2_score(y_true, y_pred)

def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return 0.0
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_model(model, X, y, n_features=None) -> dict:
    y_pred = model.predict(X)
    mae = calculate_mae(y, y_pred)
    mse = calculate_mse(y, y_pred)
    rmse = calculate_rmse(y, y_pred)
    r2 = calculate_r2(y, y_pred)
    
    adj_r2 = None
    if n_features is not None:
        adj_r2 = calculate_adjusted_r2(r2, len(y), n_features)
        
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2
    }
