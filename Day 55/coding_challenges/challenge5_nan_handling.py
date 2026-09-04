"""
===============================================================================
DAY 55 — CODING CHALLENGE 5: MISSING VALUE (NaN) HANDLING
===============================================================================
Topic: NaN Detection and NaN-Aware Aggregation Functions
Goal: Given values [10, 20, np.nan, 40, np.nan, 50], analyze missing values.
===============================================================================
"""

import numpy as np


def analyze_array_with_nans(values: np.ndarray):
    """Compute missing value count, nanmean, nanmax, and nanmin."""
    # What is used: np.isnan() boolean mask and np.nanmean / np.nanmax / np.nanmin.
    # Why it is used: Analyzes numerical arrays containing missing NaN entries cleanly.
    # How it works: Sums boolean mask to count missing items and runs nan-aware functions.
    missing_count = np.sum(np.isnan(values))
    mean_val = np.nanmean(values)
    max_val = np.nanmax(values)
    min_val = np.nanmin(values)

    return {
        "missing_count": missing_count,
        "nan_mean": mean_val,
        "nan_max": max_val,
        "nan_min": min_val,
    }


if __name__ == "__main__":
    arr = np.array([10, 20, np.nan, 40, np.nan, 50], dtype=np.float64)
    res = analyze_array_with_nans(arr)
    print("Array with NaNs Analysis Results:")
    for k, v in res.items():
        print(f"  {k:<15}: {v}")

    assert res["missing_count"] == 2, "Missing count failed"
    assert res["nan_mean"] == 30.0, "Mean failed ((10+20+40+50)/4 = 30.0)"
    assert res["nan_max"] == 50.0, "Max failed"
    assert res["nan_min"] == 10.0, "Min failed"
    print("[OK] Challenge 5 Passed Successfully!")
