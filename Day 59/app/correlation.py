"""
Module: correlation.py
Computes numerical Pearson correlation and covariance matrices across financial and quantity features.
"""

# What is used: Import pandas library.
# Why it is used: Core package for correlation and covariance matrices.
# How it works: Brings pandas namespace into scope.
import pandas as pd


CORR_NUMERIC_COLS = ["Quantity", "Unit_Price", "Discount", "Cost_Price", "Revenue", "Cost", "Profit"]


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation coefficients between numerical metrics.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Symmetric correlation matrix dataframe.
    """
    valid_cols = [col for col in CORR_NUMERIC_COLS if col in df.columns]

    # What is used: df[valid_cols].corr(numeric_only=True).
    # Why it is used: Calculates correlation coefficients r between continuous features.
    # How it works: Returns normalized linear dependence matrix between -1.0 and +1.0.
    corr_df = df[valid_cols].corr(numeric_only=True).round(4)
    return corr_df


def compute_covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise sample covariance matrix between numerical metrics.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Covariance matrix dataframe.
    """
    valid_cols = [col for col in CORR_NUMERIC_COLS if col in df.columns]

    # What is used: df[valid_cols].cov(numeric_only=True).
    # Why it is used: Measures directional co-variability between continuous variables.
    # How it works: Returns un-normalized covariance matrix.
    cov_df = df[valid_cols].cov(numeric_only=True).round(2)
    return cov_df
