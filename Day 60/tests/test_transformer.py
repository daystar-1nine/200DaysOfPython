"""
Unit Tests for app/transformer.py module.
"""

import pandas as pd
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_transform_sales_data_revenue_cost_profit(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)

    row0 = trans_df.iloc[0]
    expected_rev = round(2 * 120000.0 * 0.90, 2)  # 216000.0
    expected_cost = round(2 * 95000.0, 2)         # 190000.0
    expected_profit = round(expected_rev - expected_cost, 2)  # 26000.0

    assert row0["Revenue"] == expected_rev
    assert row0["Cost"] == expected_cost
    assert row0["Profit"] == expected_profit


def test_transform_sales_data_profit_margin(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    assert "Profit_Margin" in trans_df.columns
    assert (trans_df["Profit_Margin"] >= -100.0).all()


def test_transform_sales_data_zero_revenue_safe():
    zero_df = pd.DataFrame({
        "Order_ID": ["O1"],
        "Quantity": [0],
        "Unit_Price": [100.0],
        "Cost_Price": [50.0],
        "Discount": [0.0],
        "Order_Date": pd.to_datetime(["2026-01-01"])
    })
    trans_df = transform_sales_data(zero_df)
    assert trans_df["Profit_Margin"].iloc[0] == 0.0


def test_transform_sales_data_temporal_features(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    assert "Year" in trans_df.columns
    assert "Month" in trans_df.columns
    assert "Month_Name" in trans_df.columns
    assert "Day_Of_Week" in trans_df.columns
    assert trans_df["Year"].iloc[0] == 2026


def test_transform_sales_data_non_negative_cost(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    assert (trans_df["Cost"] >= 0).all()
