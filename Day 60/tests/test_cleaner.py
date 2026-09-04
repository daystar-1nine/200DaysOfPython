"""
Unit Tests for app/cleaner.py module.
"""

import pandas as pd
from app.cleaner import clean_sales_records


def test_clean_sales_records_deduplication(sample_raw_sales_df):
    clean_df, audit = clean_sales_records(sample_raw_sales_df)
    assert audit["duplicates_removed"] == 1
    assert len(clean_df) == 5
    assert clean_df["Order_ID"].is_unique


def test_clean_sales_records_string_trimming(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    rahul_row = clean_df[clean_df["Customer_ID"] == "C101"].iloc[0]
    assert rahul_row["Customer_Name"] == "Rahul Sharma"


def test_clean_sales_records_date_coercion(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    assert pd.api.types.is_datetime64_any_dtype(clean_df["Order_Date"])
    assert clean_df["Order_Date"].iloc[0] <= clean_df["Order_Date"].iloc[-1]


def test_clean_sales_records_numeric_nulls():
    messy_df = pd.DataFrame({
        "Order_ID": ["O1", "O2"],
        "Order_Date": ["2026-01-01", "2026-01-02"],
        "Customer_ID": ["C1", "C2"],
        "Customer_Name": ["A", "B"],
        "Region": ["North", "South"],
        "Category": ["Electronics", "Electronics"],
        "Product": ["P1", "P2"],
        "Quantity": ["5", None],
        "Unit_Price": [100.0, "200.0"],
        "Cost_Price": [50.0, 100.0],
        "Discount": [0.1, 0.0]
    })
    clean_df, audit = clean_sales_records(messy_df)
    assert audit["nulls_filled"] == 1
    assert clean_df["Quantity"].isna().sum() == 0


def test_clean_sales_records_invalid_date_fallback():
    messy_df = pd.DataFrame({
        "Order_ID": ["O1"],
        "Order_Date": ["invalid_date_string"],
        "Customer_ID": ["C1"],
        "Customer_Name": ["A"],
        "Region": ["North"],
        "Category": ["Electronics"],
        "Product": ["P1"],
        "Quantity": [1],
        "Unit_Price": [100.0],
        "Cost_Price": [50.0],
        "Discount": [0.0]
    })
    clean_df, audit = clean_sales_records(messy_df)
    assert audit["dates_coerced"] == 1
    assert clean_df["Order_Date"].iloc[0] == pd.Timestamp("2026-01-01")
