"""
Unit tests for data transformer module app/transformer.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 57 parent directory to sys.path.
import sys
from pathlib import Path

DAY57_DIR = Path(__file__).resolve().parent.parent
if str(DAY57_DIR) not in sys.path:
    sys.path.insert(0, str(DAY57_DIR))

# What is used: Import pandas and transform_data function.
# Why it is used: Asserts Revenue calculation, Order_Date datetime parsing, Month period extraction, and Discount_Range binning.
# How it works: Executes transform_data on clean test DataFrame.
import pandas as pd
from app.transformer import transform_data


def test_transform_revenue_calculation():
    """
    Test Revenue calculation formula: Quantity * Unit_Price * (1 - Discount).
    """
    # What is used: Input DataFrame with known Quantity, Unit_Price, and Discount.
    # Why it is used: Verifies net revenue calculation accuracy.
    # How it works: Compares calculated Revenue column against expected float values.
    test_df = pd.DataFrame({
        "Order_ID": [101, 102],
        "Order_Date": ["2026-01-05", "2026-02-10"],
        "Customer_ID": ["C1", "C2"],
        "Customer_Name": ["A", "B"],
        "Region": ["West", "East"],
        "Category": ["Electronics", "Furniture"],
        "Product": ["Laptop", "Chair"],
        "Quantity": [2, 5],
        "Unit_Price": [60000.0, 4000.0],
        "Discount": [0.05, 0.10]
    })

    transformed = transform_data(test_df)
    # Row 0: 2 * 60000 * 0.95 = 114000.0
    # Row 1: 5 * 4000 * 0.90 = 18000.0
    assert transformed.loc[0, "Revenue"] == 114000.0
    assert transformed.loc[1, "Revenue"] == 18000.0


def test_transform_datetime_and_month_extraction():
    """
    Test Order_Date datetime parsing and Month period string extraction.
    """
    # What is used: Input DataFrame with string dates.
    # Why it is used: Verifies datetime parsing and Month column creation.
    # How it works: Asserts Order_Date is datetime64 and Month equals '2026-01' and '2026-02'.
    test_df = pd.DataFrame({
        "Order_ID": [101, 102],
        "Order_Date": ["2026-01-05", "2026-02-10"],
        "Customer_ID": ["C1", "C2"],
        "Customer_Name": ["A", "B"],
        "Region": ["West", "East"],
        "Category": ["Electronics", "Furniture"],
        "Product": ["Laptop", "Chair"],
        "Quantity": [1, 1],
        "Unit_Price": [100.0, 200.0],
        "Discount": [0.0, 0.0]
    })

    transformed = transform_data(test_df)
    assert pd.api.types.is_datetime64_any_dtype(transformed["Order_Date"])
    assert transformed.loc[0, "Month"] == "2026-01"
    assert transformed.loc[1, "Month"] == "2026-02"


def test_transform_discount_range_binning():
    """
    Test Discount_Range categorical binning logic.
    """
    # What is used: Input DataFrame with various discount rates.
    # Why it is used: Verifies pd.cut binning categories.
    # How it works: Asserts assigned Discount_Range labels for 0%, 5%, 15%, 25%.
    test_df = pd.DataFrame({
        "Order_ID": [1, 2, 3, 4],
        "Order_Date": ["2026-01-01"] * 4,
        "Customer_ID": ["C1"] * 4,
        "Customer_Name": ["A"] * 4,
        "Region": ["West"] * 4,
        "Category": ["Electronics"] * 4,
        "Product": ["P1"] * 4,
        "Quantity": [1] * 4,
        "Unit_Price": [100.0] * 4,
        "Discount": [0.00, 0.05, 0.15, 0.25]
    })

    transformed = transform_data(test_df)
    assert transformed.loc[0, "Discount_Range"] == "No Discount (0%)"
    assert transformed.loc[1, "Discount_Range"] == "Low (1-9%)"
    assert transformed.loc[2, "Discount_Range"] == "Medium (10-19%)"
    assert transformed.loc[3, "Discount_Range"] == "High (20%+)"
