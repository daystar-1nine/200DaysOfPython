"""
Core statistical calculation functions.
"""
import numpy as np
try:
    from app.validator import validate_numeric_array
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array

def compute_descriptive_stats(data: np.ndarray) -> dict:
    arr = validate_numeric_array(data)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    q1_val = float(np.percentile(arr, 25))
    q3_val = float(np.percentile(arr, 75))
    iqr_val = float(q3_val - q1_val)
    se_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
    
    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std_dev": round(std_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "q1": round(q1_val, 4),
        "q3": round(q3_val, 4),
        "iqr": round(iqr_val, 4),
        "standard_error": round(se_val, 4)
    }
