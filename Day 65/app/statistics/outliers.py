"""
IQR-based outlier detection module.
"""

import pandas as pd
import numpy as np


def detect_iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> dict:
    clean_s = series.dropna()
    if clean_s.empty:
        return {
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "count": 0,
            "percentage": 0.0,
            "outliers": []
        }
        
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    iqr_val = q3 - q1
    
    lower_bound = q1 - multiplier * iqr_val
    upper_bound = q3 + multiplier * iqr_val
    
    mask = (clean_s < lower_bound) | (clean_s > upper_bound)
    outlier_series = clean_s[mask]
    
    return {
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "count": int(mask.sum()),
        "percentage": round((mask.sum() / len(clean_s)) * 100.0, 2),
        "outliers": outlier_series.tolist()
    }
