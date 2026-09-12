import numpy as np

def calculate_residuals(y_true, y_pred) -> np.ndarray:
    return np.array(y_true) - np.array(y_pred)

def residual_stats(residuals) -> dict:
    from scipy.stats import skew
    mean_res = np.mean(residuals)
    return {
        "mean": float(mean_res),
        "std": float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "skewness": float(skew(residuals)),
        "is_zero_mean": abs(mean_res) < 1e-7
    }

def analyze_residuals(y_true, y_pred) -> dict:
    residuals = calculate_residuals(y_true, y_pred)
    return residual_stats(residuals)
