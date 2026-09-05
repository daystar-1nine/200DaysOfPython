"""
Statistical Analytics & Metrics Module
======================================
Computes parametric and non-parametric summary statistics, distribution moments,
IQR bounds, and pairwise Pearson correlations.
"""

import pandas as pd
import numpy as np

def compute_univariate_stats(series: pd.Series) -> dict:
    """
    Calculates summary statistics and shape moments for a continuous variable.

    # What is used: Mean, median, std, quantiles, skewness, kurtosis, and IQR
    # Why it is used: Quantifies central tendency, dispersion, and tail asymmetry
    # How it works: Uses Pandas and SciPy formulas on clean series
    """
    clean_s = series.dropna()
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    iqr = q3 - q1

    return {
        "count": int(clean_s.count()),
        "mean": float(clean_s.mean()),
        "median": float(clean_s.median()),
        "std": float(clean_s.std()),
        "min": float(clean_s.min()),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "max": float(clean_s.max()),
        "skewness": float(clean_s.skew()),
        "kurtosis": float(clean_s.kurtosis()),
        "outlier_threshold_low": q1 - 1.5 * iqr,
        "outlier_threshold_high": q3 + 1.5 * iqr
    }

def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates aggregate financial metrics by product category.
    """
    summary = df.groupby("Category").agg(
        Order_Count=("Order_ID", "count"),
        Total_Revenue=("Revenue", "sum"),
        Mean_Revenue=("Revenue", "mean"),
        Median_Revenue=("Revenue", "median"),
        Total_Profit=("Profit", "sum"),
        Mean_Profit_Margin=("Profit_Margin", "mean")
    ).reset_index()

    return summary

def compute_correlation_matrix(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix for specified numerical columns.
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns]
    return numeric_df.corr(method="pearson")

def extract_extreme_correlations(corr_df: pd.DataFrame, top_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts top positive and top negative correlation pairs excluding identity diagonal.
    """
    pairs = []
    cols = corr_df.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            r_val = corr_df.loc[col1, col2]
            pairs.append({"Feature_1": col1, "Feature_2": col2, "Pearson_R": float(r_val)})

    df_pairs = pd.DataFrame(pairs)
    top_positive = df_pairs.sort_values("Pearson_R", ascending=False).head(top_n).reset_index(drop=True)
    top_negative = df_pairs.sort_values("Pearson_R", ascending=True).head(top_n).reset_index(drop=True)

    return top_positive, top_negative
