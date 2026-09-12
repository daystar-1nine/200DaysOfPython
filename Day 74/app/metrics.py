import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def calculate_mse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred))

def calculate_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(calculate_mse(y_true, y_pred)))

def calculate_r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))

def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    if n - p - 1 <= 0:
        return float('nan')
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

adjusted_r2 = calculate_adjusted_r2

def calculate_metrics(y_true, y_pred, n_features: int) -> dict:
    n_samples = len(y_true)
    mae = calculate_mae(y_true, y_pred)
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    adj_r2 = calculate_adjusted_r2(r2, n_samples, n_features)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Adjusted_R2': adj_r2,
        'n_samples': n_samples,
        'n_features': n_features
    }

def format_metrics_report(metrics_or_y_true, y_pred=None, n=None, p=None) -> str:
    if isinstance(metrics_or_y_true, dict):
        metrics = metrics_or_y_true
    else:
        y_true = metrics_or_y_true
        n_feat = p if p is not None else 1
        metrics = calculate_metrics(y_true, y_pred, n_features=n_feat)
        if n is not None:
            metrics['n_samples'] = n
        if p is not None:
            metrics['n_features'] = p
            metrics['Adjusted_R2'] = calculate_adjusted_r2(metrics['R2'], metrics['n_samples'], metrics['n_features'])

    return f"""
Model Performance Metrics:
--------------------------
R-squared:      {metrics.get('R2', 0):.4f}
Adjusted R2:    {metrics.get('Adjusted_R2', 0):.4f}
RMSE:           {metrics.get('RMSE', 0):.4f}
MAE:            {metrics.get('MAE', 0):.4f}
MSE:            {metrics.get('MSE', 0):.4f}
Samples (n):    {metrics.get('n_samples', 0)}
Features (p):   {metrics.get('n_features', 0)}
"""
