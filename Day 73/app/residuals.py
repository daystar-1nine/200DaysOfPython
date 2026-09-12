"""
Day 73 - Residual Diagnostics Engine
Calculates residuals, verifies zero-mean property, and checks homoscedasticity.
"""
from typing import Dict, Any
import numpy as np

def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    
    residuals = y_t - y_p
    n = len(residuals)
    
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals, ddof=1))
    sum_res = float(np.sum(residuals))
    
    # Skewness
    skew = float(np.mean(((residuals - mean_res) / (std_res + 1e-12)) ** 3))
    
    # Variance ratio across median split of predictions (Homoscedasticity check)
    sorted_indices = np.argsort(y_p)
    res_sorted = residuals[sorted_indices]
    half = n // 2
    var_lower = float(np.var(res_sorted[:half], ddof=1))
    var_upper = float(np.var(res_sorted[half:], ddof=1))
    variance_ratio = float(var_upper / (var_lower + 1e-12))
    
    is_homoscedastic = 0.5 <= variance_ratio <= 2.0
    
    return {
        "residuals": residuals,
        "mean_residual": mean_res,
        "std_residual": std_res,
        "sum_residual": sum_res,
        "skewness": skew,
        "variance_ratio": variance_ratio,
        "is_homoscedastic": is_homoscedastic
    }
