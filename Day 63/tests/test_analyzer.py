"""
Tests for Statistical Analytics Module
======================================
"""

import pandas as pd
import numpy as np
from app.analyzer import (
    compute_univariate_stats,
    compute_category_summary,
    compute_correlation_matrix,
    extract_extreme_correlations
)

def test_compute_univariate_stats_keys():
    s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    stats = compute_univariate_stats(s)
    expected_keys = ["count", "mean", "median", "std", "min", "q1", "q3", "iqr", "max", "skewness", "kurtosis"]
    for k in expected_keys:
        assert k in stats
    assert stats["count"] == 10
    assert stats["mean"] == 55.0
    assert stats["median"] == 55.0

def test_compute_univariate_stats_skewness():
    # Heavily right skewed distribution
    s = pd.Series([10, 12, 11, 13, 12, 14, 10, 15, 100, 250])
    stats = compute_univariate_stats(s)
    assert stats["skewness"] > 1.0

def test_compute_category_summary(clean_sample_df):
    summary = compute_category_summary(clean_sample_df)
    assert isinstance(summary, pd.DataFrame)
    assert "Category" in summary.columns
    assert "Total_Revenue" in summary.columns
    assert "Order_Count" in summary.columns
    assert len(summary) == clean_sample_df["Category"].nunique()

def test_compute_correlation_matrix(clean_sample_df):
    cols = ["Quantity", "Revenue", "Profit", "Profit_Margin"]
    corr = compute_correlation_matrix(clean_sample_df, cols)
    assert corr.shape == (4, 4)
    # Diagonal must equal 1.0
    for col in cols:
        assert abs(corr.loc[col, col] - 1.0) < 1e-6
    # Symmetry r_ij == r_ji
    assert abs(corr.loc["Revenue", "Profit"] - corr.loc["Profit", "Revenue"]) < 1e-6

def test_extract_extreme_correlations(clean_sample_df):
    cols = ["Quantity", "Revenue", "Profit", "Profit_Margin"]
    corr = compute_correlation_matrix(clean_sample_df, cols)
    top_pos, top_neg = extract_extreme_correlations(corr, top_n=2)
    assert len(top_pos) == 2
    assert len(top_neg) == 2
    assert top_pos.iloc[0]["Pearson_R"] >= top_pos.iloc[1]["Pearson_R"]
    assert top_neg.iloc[0]["Pearson_R"] <= top_neg.iloc[1]["Pearson_R"]
