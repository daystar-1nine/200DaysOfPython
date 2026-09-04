"""
Unit Tests for app/analysis/time_series.py module.
"""

import pandas as pd
from app.analysis.time_series import analyze_daily_series, analyze_monthly_series
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_analyze_monthly_series_metrics(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    monthly_df = analyze_monthly_series(trans_df)

    assert "Year_Month" in monthly_df.columns
    assert "total_revenue" in monthly_df.columns
    assert "MoM_Growth_%" in monthly_df.columns
    assert "Rolling_3M_Avg" in monthly_df.columns


def test_analyze_monthly_series_sorted(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    monthly_df = analyze_monthly_series(trans_df)

    months = monthly_df["Year_Month"].tolist()
    assert months == sorted(months)


def test_analyze_daily_series(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    daily_df = analyze_daily_series(trans_df)

    assert "daily_revenue" in daily_df.columns
    assert "7_Day_Rolling_Avg" in daily_df.columns
    assert len(daily_df) == len(trans_df["Order_Date"].unique())
