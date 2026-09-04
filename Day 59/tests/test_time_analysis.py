"""
Unit Tests for app/time_analysis.py module.
"""

import pandas as pd
from app.cleaner import clean_sales_data
from app.time_analysis import analyze_daily_rolling_trends, analyze_monthly_trends
from app.transformer import compute_derived_metrics


def test_analyze_monthly_trends(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    monthly_df = analyze_monthly_trends(enriched_df)

    assert "Year_Month" in monthly_df.columns
    assert "MoM_Growth_%" in monthly_df.columns
    assert "Rolling_3M_Avg" in monthly_df.columns
    assert len(monthly_df) >= 1


def test_analyze_daily_rolling_trends(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    daily_df = analyze_daily_rolling_trends(enriched_df)

    assert "daily_revenue" in daily_df.columns
    assert "7_Day_Rolling_Avg" in daily_df.columns
    assert len(daily_df) == len(enriched_df["Order_Date"].unique())


def test_analyze_monthly_trends_diff(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    monthly_df = analyze_monthly_trends(enriched_df)

    assert "Revenue_Diff" in monthly_df.columns
    assert "Prev_Month_Revenue" in monthly_df.columns
    # First month previous revenue must be NaN
    assert pd.isna(monthly_df["Prev_Month_Revenue"].iloc[0])


def test_analyze_monthly_trends_sorted(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    monthly_df = analyze_monthly_trends(enriched_df)

    # Months should be chronologically ordered
    months = monthly_df["Year_Month"].tolist()
    assert months == sorted(months)
