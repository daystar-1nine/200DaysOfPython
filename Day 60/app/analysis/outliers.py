"""
Module: outliers.py
Detects statistical outliers using Interquartile Range (IQR) boundaries for numerical features.
"""

# What is used: Import pandas library.
# Why it is used: Quantile calculations and boolean outlier masking.
# How it works: Derives Q1, Q3, IQR, and lower/upper bounds for target columns.
import pandas as pd

OUTLIER_COLS = ["Revenue", "Profit", "Quantity", "Unit_Price"]


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    Perform IQR outlier audit across key numerical transaction columns.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        dict: Outlier report dictionary mapping column names to Q1, Q3, IQR, bounds, counts, and outlier DataFrames.
    """
    results = {}
    for col in OUTLIER_COLS:
        if col in df.columns:
            series = df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # What is used: Boolean mask filtering.
            # Why it is used: Filters rows exceeding IQR bounds.
            # How it works: Evaluates (val < lower_bound) | (val > upper_bound).
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_df = df[mask].copy()

            results[col] = {
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": len(outlier_df),
                "outliers": outlier_df
            }
    return results
