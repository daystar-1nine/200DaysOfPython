"""
Unit Tests for app/analysis/regional.py module.
"""

from app.analysis.regional import analyze_regions
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_analyze_regions_rankings(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    reg_df = analyze_regions(trans_df)

    assert "Rank" in reg_df.columns
    assert reg_df["Rank"].iloc[0] == 1


def test_analyze_regions_revenue_share(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    reg_df = analyze_regions(trans_df)

    assert "Revenue_Share_%" in reg_df.columns
    total_share = reg_df["Revenue_Share_%"].sum()
    assert 99.0 <= total_share <= 101.0


def test_analyze_regions_metrics(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    reg_df = analyze_regions(trans_df)

    assert "total_revenue" in reg_df.columns
    assert "total_profit" in reg_df.columns
    assert "order_count" in reg_df.columns
