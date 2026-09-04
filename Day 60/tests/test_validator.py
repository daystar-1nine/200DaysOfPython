"""
Unit Tests for app/validator.py module.
"""

import pandas as pd
from app.cleaner import clean_sales_records
from app.validator import validate_sales_data


def test_validate_sales_data_success(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    results = validate_sales_data(clean_df)
    assert results["is_valid"] is True
    assert results["rules"]["unique_order_ids"]["passed"] is True
    assert results["rules"]["positive_quantity"]["passed"] is True


def test_validate_sales_data_duplicate_order_id():
    bad_df = pd.DataFrame({
        "Order_ID": ["O1", "O1"],
        "Quantity": [1, 2],
        "Unit_Price": [100.0, 200.0],
        "Cost_Price": [50.0, 100.0],
        "Discount": [0.0, 0.1],
        "Region": ["North", "South"],
        "Order_Date": pd.to_datetime(["2026-01-01", "2026-01-02"])
    })
    res = validate_sales_data(bad_df)
    assert res["is_valid"] is False
    assert res["rules"]["unique_order_ids"]["passed"] is False


def test_validate_sales_data_negative_quantity():
    bad_df = pd.DataFrame({
        "Order_ID": ["O1"],
        "Quantity": [-5],
        "Unit_Price": [100.0],
        "Cost_Price": [50.0],
        "Discount": [0.0],
        "Region": ["North"],
        "Order_Date": pd.to_datetime(["2026-01-01"])
    })
    res = validate_sales_data(bad_df)
    assert res["rules"]["positive_quantity"]["passed"] is False


def test_validate_sales_data_invalid_discount():
    bad_df = pd.DataFrame({
        "Order_ID": ["O1"],
        "Quantity": [2],
        "Unit_Price": [100.0],
        "Cost_Price": [50.0],
        "Discount": [1.5],
        "Region": ["North"],
        "Order_Date": pd.to_datetime(["2026-01-01"])
    })
    res = validate_sales_data(bad_df)
    assert res["rules"]["valid_discount_range"]["passed"] is False


def test_validate_sales_data_invalid_region():
    bad_df = pd.DataFrame({
        "Order_ID": ["O1"],
        "Quantity": [2],
        "Unit_Price": [100.0],
        "Cost_Price": [50.0],
        "Discount": [0.1],
        "Region": ["Antarctica"],
        "Order_Date": pd.to_datetime(["2026-01-01"])
    })
    res = validate_sales_data(bad_df)
    assert res["rules"]["valid_regions"]["passed"] is False
