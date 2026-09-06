"""
Challenge 2 — IQR Outlier Detector
Implements detect_outliers(series) returning bounds, count, percentage, and masks.
"""

import pandas as pd
import numpy as np


def detect_outliers(series: pd.Series, multiplier: float = 1.5) -> dict:
    clean_s = series.dropna()
    if clean_s.empty:
        return {
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "outlier_values": []
        }
    
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    iqr_val = q3 - q1
    
    lower_bound = q1 - multiplier * iqr_val
    upper_bound = q3 + multiplier * iqr_val
    
    mask = (clean_s < lower_bound) | (clean_s > upper_bound)
    outliers = clean_s[mask].tolist()
    count = len(outliers)
    percentage = (count / len(clean_s)) * 100.0
    
    return {
        "outlier_count": count,
        "outlier_percentage": round(percentage, 2),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_values": outliers
    }


if __name__ == "__main__":
    data = pd.Series([12, 14, 15, 16, 17, 18, 19, 20, 22, 25, 95, -10])
    results = detect_outliers(data)
    print("=== CHALLENGE 2: IQR OUTLIER DETECTION ===")
    for k, v in results.items():
        print(f"  {k:<20}: {v}")
