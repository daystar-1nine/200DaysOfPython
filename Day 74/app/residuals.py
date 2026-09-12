import numpy as np

from scipy.stats import skew

def calculate_residuals(y_true, y_pred) -> np.ndarray:
    return np.asarray(y_true) - np.asarray(y_pred)

def residual_stats(residuals) -> dict:
    residuals = np.asarray(residuals)
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    median_res = float(np.median(residuals))
    skew_res = float(skew(residuals)) if len(residuals) > 0 else 0.0
    is_zero_mean = abs(mean_res) < 0.1 * std_res if std_res > 0 else True

    return {
        'mean': mean_res,
        'std': std_res,
        'median': median_res,
        'skewness': skew_res,
        'is_zero_mean': is_zero_mean
    }

def analyze_residuals(y_true, y_pred) -> dict:
    residuals = calculate_residuals(y_true, y_pred)
    stats = residual_stats(residuals)
    
    # Homoscedasticity ratio
    sorted_indices = np.argsort(y_pred)
    sorted_res = residuals[sorted_indices]
    half = len(sorted_res) // 2
    if half > 0:
        std_h1 = np.std(sorted_res[:half])
        std_h2 = np.std(sorted_res[half:])
        ratio = std_h1 / std_h2 if std_h2 != 0 else 1.0
    else:
        ratio = 1.0

    stats['homoscedasticity_ratio'] = float(ratio)
    return stats

def plot_residuals_data(y_true, y_pred) -> tuple:
    residuals = calculate_residuals(y_true, y_pred)
    return np.asarray(y_pred), residuals
