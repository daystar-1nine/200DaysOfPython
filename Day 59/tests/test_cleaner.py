"""
Unit Tests for app/cleaner.py module.
"""

import pandas as pd
from app.cleaner import clean_sales_data


def test_clean_sales_data_deduplication(sample_raw_sales_df):
    clean_df, stats = clean_sales_data(sample_raw_sales_df)
    assert stats["duplicates_removed"] == 1
    assert len(clean_df) == 5
    assert clean_df["Order_ID"].is_unique


def test_clean_sales_data_string_trimming(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    rahul_row = clean_df[clean_df["Customer_ID"] == "C101"].iloc[0]
    assert rahul_row["Customer_Name"] == "Rahul Sharma"


def test_clean_sales_data_datetime_conversion(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    assert pd.api.types.is_datetime64_any_dtype(clean_df["Order_Date"])
    assert clean_df["Order_Date"].iloc[0] <= clean_df["Order_Date"].iloc[-1]
