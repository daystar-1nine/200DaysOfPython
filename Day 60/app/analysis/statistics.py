"""
Module: statistics.py
Computes descriptive statistics, quantiles, percentiles, correlation, and covariance matrices for numerical features.
"""

# What is used: Import pandas library.
# Why it is used: Computes distribution metrics, correlation, and covariance matrices.
# How it works: Calculates mean, std, quantiles, corr(), and cov() on numerical columns.
import pandas as pd

STATS_COLS = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Cost", "Profit"]


def compute_numerical_statistics(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics and percentiles for numerical variables.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        dict: Statistical summary mapped by column name.
    """
    summary = {}
    for col in STATS_COLS:
        if col in df.columns:
            series = df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            summary[col] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q1": round(q1, 2),
                "median": round(float(series.median()), 2),
                "q3": round(q3, 2),
                "max": round(float(series.max()), 2),
                "iqr": round(iqr, 2)
            }
    return summary


def compute_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation and sample covariance matrices across numerical features.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (correlation_matrix, covariance_matrix)
    """
    valid_cols = [c for c in STATS_COLS if c in df.columns]
    corr_df = df[valid_cols].corr(numeric_only=True).round(4)
    cov_df = df[valid_cols].cov(numeric_only=True).round(2)
    return corr_df, cov_df
