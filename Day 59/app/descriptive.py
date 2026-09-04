"""
Module: descriptive.py
Computes descriptive statistics, quantiles, percentiles, and IQR metric summaries for numerical variables.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core libraries for distribution metrics and percentile calculations.
# How it works: Brings pandas and numpy namespaces into scope.
import numpy as np
import pandas as pd


NUMERIC_METRIC_COLS = ["Quantity", "Unit_Price", "Discount", "Cost_Price", "Revenue", "Cost", "Profit"]


def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive descriptive statistics (mean, median, std, min, max, Q1, Q3, IQR) for numerical variables.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        dict: Descriptive statistics dictionary mapped by column name.
    """
    stats_summary = {}

    for col in NUMERIC_METRIC_COLS:
        if col in df.columns:
            series = df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            stats_summary[col] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q1_25%": round(q1, 2),
                "median_50%": round(float(series.median()), 2),
                "q3_75%": round(q3, 2),
                "max": round(float(series.max()), 2),
                "iqr": round(iqr, 2)
            }

    return stats_summary
