"""
Module: outliers.py
Detects statistical outliers using Interquartile Range (IQR) bounds for Revenue, Profit, and Quantity features.
"""

# What is used: Import pandas library.
# Why it is used: Quantile computations and boolean outlier masking.
# How it works: Brings pandas namespace into scope.
import pandas as pd


OUTLIER_TARGET_COLS = ["Revenue", "Profit", "Quantity", "Unit_Price"]


def detect_all_outliers(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive IQR-based outlier audit across target numerical columns.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        dict: Outlier metrics dictionary mapping column names to Q1, Q3, IQR, bounds, count, and outlier DataFrames.
    """
    audit_results = {}

    for col in OUTLIER_TARGET_COLS:
        if col in df.columns:
            series = df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # What is used: Boolean mask filtering.
            # Why it is used: Identifies rows outside valid IQR statistical boundaries.
            # How it works: Masks values strictly < lower_bound or > upper_bound.
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_df = df[mask].copy()

            audit_results[col] = {
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": len(outlier_df),
                "outliers": outlier_df
            }

    return audit_results
