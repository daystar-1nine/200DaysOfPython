"""
Challenge 1 — Comprehensive Statistical Summary Function
Implements statistical_summary(series) returning a 13-metric dictionary.
"""

import pandas as pd
import numpy as np


def statistical_summary(series: pd.Series) -> dict:
    clean_s = series.dropna()
    if clean_s.empty:
        raise ValueError("Series is empty or contains only NaN values.")
    
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    iqr_val = q3 - q1
    min_val = float(clean_s.min())
    max_val = float(clean_s.max())
    
    modes = clean_s.mode().tolist()
    primary_mode = modes[0] if len(modes) == 1 else modes

    return {
        "mean": float(clean_s.mean()),
        "median": float(clean_s.median()),
        "mode": primary_mode,
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "variance": float(clean_s.var(ddof=1)),
        "std": float(clean_s.std(ddof=1)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr_val,
        "skewness": float(clean_s.skew()),
        "kurtosis": float(clean_s.kurt()),
    }


if __name__ == "__main__":
    test_data = pd.Series([10, 20, 25, 30, 35, 40, 45, 50, 50, 100])
    summary = statistical_summary(test_data)
    print("=== CHALLENGE 1: STATISTICAL SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.4f}")
        else:
            print(f"  {k:<12}: {v}")
