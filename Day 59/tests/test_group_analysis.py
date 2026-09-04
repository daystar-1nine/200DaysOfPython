"""
Unit Tests for app/group_analysis.py module.
"""

from app.cleaner import clean_sales_data
from app.group_analysis import (
    analyze_category_performance,
    analyze_customer_performance,
    analyze_product_performance,
    analyze_regional_performance,
)
from app.transformer import compute_derived_metrics


def test_analyze_regional_performance(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    reg_df = analyze_regional_performance(enriched_df)

    assert "total_revenue" in reg_df.columns
    assert "Revenue_Rank" in reg_df.columns
    assert reg_df["Revenue_Rank"].iloc[0] == 1


def test_analyze_category_performance(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    cat_df = analyze_category_performance(enriched_df)

    assert "total_revenue" in cat_df.columns
    assert "total_profit" in cat_df.columns
    assert len(cat_df) <= len(enriched_df["Category"].unique())


def test_analyze_product_performance_rankings(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    top_rev, top_qty, top_prof = analyze_product_performance(enriched_df)

    assert len(top_rev) <= 10
    assert "Overall_Rank" in top_rev.columns
    assert "Category_Rank" in top_rev.columns


def test_analyze_customer_performance(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    cust_df = analyze_customer_performance(enriched_df)

    assert "Customer_Rank" in cust_df.columns
    assert "Above_Average_Customer" in cust_df.columns
    assert cust_df["Customer_Rank"].iloc[0] == 1


def test_analyze_customer_performance_aov(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    cust_df = analyze_customer_performance(enriched_df)

    assert (cust_df["aov"] > 0).all()
    assert len(cust_df) == len(enriched_df["Customer_ID"].unique())


def test_analyze_category_performance_discount(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    cat_df = analyze_category_performance(enriched_df)

    assert "avg_discount" in cat_df.columns
    assert (cat_df["avg_discount"] >= 0.0).all()
