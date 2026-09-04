"""
Unit Tests for app/analysis/customer.py module.
"""

from app.analysis.customer import analyze_customers
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_analyze_customers_rankings(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    cust_df = analyze_customers(trans_df)

    assert "Rank" in cust_df.columns
    assert cust_df["Rank"].iloc[0] == 1
    assert "total_revenue" in cust_df.columns


def test_analyze_customers_aov(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    cust_df = analyze_customers(trans_df)

    assert "aov" in cust_df.columns
    assert (cust_df["aov"] > 0).all()


def test_analyze_customers_above_average_flag(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    cust_df = analyze_customers(trans_df)

    assert "Above_Average_Spend" in cust_df.columns
    assert cust_df["Above_Average_Spend"].dtype == bool
