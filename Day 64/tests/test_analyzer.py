"""
Tests for Pure Business Analytics Engine
========================================
"""

import pandas as pd
import numpy as np
from app.analyzer import (
    get_dataset_overview,
    get_distribution_stats,
    get_regional_revenue_stats,
    get_category_profit_stats,
    get_region_counts,
    get_region_category_revenue,
    get_monthly_trends,
    get_correlation_matrix,
    get_top_customers,
    generate_15_business_insights
)

def test_get_dataset_overview(clean_sample_df):
    overview = get_dataset_overview(clean_sample_df)
    expected_keys = ["rows", "columns", "unique_customers", "unique_products", "unique_categories", "unique_regions", "total_revenue", "total_profit"]
    for k in expected_keys:
        assert k in overview
    assert overview["rows"] == len(clean_sample_df)

def test_get_distribution_stats(clean_sample_df):
    stats = get_distribution_stats(clean_sample_df, "Revenue")
    for k in ["mean", "median", "std", "iqr", "skewness", "kurtosis"]:
        assert k in stats
    assert stats["count"] == len(clean_sample_df)

def test_get_regional_revenue_stats(clean_sample_df):
    res = get_regional_revenue_stats(clean_sample_df)
    assert isinstance(res, pd.DataFrame)
    assert "Mean_Revenue" in res.columns
    assert len(res) == clean_sample_df["Region"].nunique()

def test_get_category_profit_stats(clean_sample_df):
    res = get_category_profit_stats(clean_sample_df)
    assert "Profit_Margin_Pct" in res.columns
    assert len(res) == clean_sample_df["Category"].nunique()

def test_get_region_counts(clean_sample_df):
    res = get_region_counts(clean_sample_df)
    assert res["Order_Count"].sum() == len(clean_sample_df)

def test_get_region_category_revenue(clean_sample_df):
    res = get_region_category_revenue(clean_sample_df)
    assert "Region" in res.columns
    assert "Category" in res.columns
    assert "Revenue" in res.columns

def test_get_monthly_trends(clean_sample_df):
    res = get_monthly_trends(clean_sample_df)
    assert "Month" in res.columns
    assert "Revenue" in res.columns

def test_get_correlation_matrix(clean_sample_df):
    cols = ["Quantity", "Revenue", "Profit"]
    corr = get_correlation_matrix(clean_sample_df, cols)
    assert corr.shape == (3, 3)
    for c in cols:
        assert abs(corr.loc[c, c] - 1.0) < 1e-6

def test_get_top_customers(clean_sample_df):
    top_rev, top_prof, summary = get_top_customers(clean_sample_df, 5)
    assert len(top_rev) <= 5
    assert len(top_prof) <= 5
    assert "AOV" in top_rev.columns
    assert "average_order_value" in summary

def test_generate_15_business_insights(clean_sample_df):
    insights = generate_15_business_insights(clean_sample_df)
    assert len(insights) >= 15
    for ins in insights:
        assert "number" in ins
        assert "title" in ins
        assert "observation" in ins
        assert "evidence" in ins
        assert "implication" in ins
