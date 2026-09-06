"""
Z-score standardization and anomaly screening module.
"""

import pandas as pd
import numpy as np


def calculate_zscores(series: pd.Series, ddof: int = 1) -> pd.Series:
    clean_s = series.dropna()
    if len(clean_s) <= 1:
        return pd.Series([0.0] * len(clean_s), index=clean_s.index)
        
    mean_val = clean_s.mean()
    std_val = clean_s.std(ddof=ddof)
    
    if std_val == 0 or np.isnan(std_val):
        return pd.Series([0.0] * len(clean_s), index=clean_s.index)
        
    return (clean_s - mean_val) / std_val


def detect_zscore_outliers(series: pd.Series, threshold: float = 3.0, ddof: int = 1) -> dict:
    clean_s = series.dropna()
    if clean_s.empty:
        return {
            "threshold": threshold,
            "count": 0,
            "percentage": 0.0,
            "outliers": []
        }
        
    z_scores = calculate_zscores(clean_s, ddof=ddof)
    mask = np.abs(z_scores) > threshold
    outlier_vals = clean_s[mask].tolist()
    
    return {
        "threshold": threshold,
        "count": int(mask.sum()),
        "percentage": round((mask.sum() / len(clean_s)) * 100.0, 2),
        "outliers": outlier_vals
    }
