"""
Unit tests for data cleaner module app/cleaner.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 57 parent directory to sys.path.
import sys
from pathlib import Path

DAY57_DIR = Path(__file__).resolve().parent.parent
if str(DAY57_DIR) not in sys.path:
    sys.path.insert(0, str(DAY57_DIR))

# What is used: Import pandas and clean_data function.
# Why it is used: Tests deduplication, NaN imputation, string whitespace trimming, and range filtering.
# How it works: Executes clean_data on test DataFrames.
import pandas as pd
from app.cleaner import clean_data


def test_clean_duplicates_removed(sample_raw_sales_df):
    """
    Test duplicate Order_ID records are removed.
    """
    # What is used: clean_data execution on sample raw DataFrame with duplicate Order_ID 1001.
    # Why it is used: Confirms duplicate removal counter and resulting row count.
    # How it works: Asserts duplicates_removed stat is 1.
    cleaned_df, stats = clean_data(sample_raw_sales_df)
    assert stats["duplicates_removed"] == 1
    assert 1001 in cleaned_df["Order_ID"].tolist()


def test_clean_null_imputation(sample_raw_sales_df):
    """
    Test missing numerical values are imputed with subject median.
    """
    # What is used: clean_data execution on sample raw DataFrame containing NaN Quantity.
    # Why it is used: Asserts nulls_filled stat is > 0 and 0 NaNs remain in numeric columns.
    # How it works: Checks null count across Quantity, Unit_Price, Discount columns.
    cleaned_df, stats = clean_data(sample_raw_sales_df)
    assert stats["nulls_filled"] > 0
    assert cleaned_df[["Quantity", "Unit_Price", "Discount"]].isnull().sum().sum() == 0


def test_clean_string_stripping(sample_raw_sales_df):
    """
    Test accidental leading/trailing whitespace is stripped from text columns.
    """
    # What is used: clean_data execution on whitespace-padded string fields.
    # Why it is used: Asserts string fields are cleanly trimmed.
    # How it works: Compares Customer_ID and Customer_Name string values.
    cleaned_df, _ = clean_data(sample_raw_sales_df)
    cust_id_1001 = cleaned_df.loc[cleaned_df["Order_ID"] == 1001, "Customer_ID"].values[0]
    cust_name_1001 = cleaned_df.loc[cleaned_df["Order_ID"] == 1001, "Customer_Name"].values[0]
    assert cust_id_1001 == "C001"
    assert cust_name_1001 == "Rahul Sharma"


def test_clean_invalid_quantity_dropped():
    """
    Test rows with Quantity <= 0 are dropped.
    """
    # What is used: Test DataFrame with negative and zero quantities.
    # Why it is used: Verifies numerical range validation for Quantity.
    # How it works: Asserts invalid_rows_dropped stat increases and invalid rows are removed.
    invalid_df = pd.DataFrame({
        "Order_ID": [1, 2, 3],
        "Order_Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Customer_ID": ["C1", "C2", "C3"],
        "Customer_Name": ["A", "B", "C"],
        "Region": ["West", "East", "South"],
        "Category": ["Electronics", "Electronics", "Furniture"],
        "Product": ["P1", "P2", "P3"],
        "Quantity": [2, -1, 0],
        "Unit_Price": [100.0, 200.0, 300.0],
        "Discount": [0.0, 0.1, 0.0]
    })
    cleaned_df, stats = clean_data(invalid_df)
    assert stats["invalid_rows_dropped"] == 2
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["Order_ID"] == 1


def test_clean_invalid_price_and_discount_dropped():
    """
    Test rows with Unit_Price <= 0 or Discount < 0 or Discount > 1 are dropped.
    """
    # What is used: Test DataFrame with invalid price (-50) and invalid discount (1.5).
    # Why it is used: Verifies boundary filtering for Unit_Price and Discount.
    # How it works: Asserts invalid rows are filtered out.
    invalid_df = pd.DataFrame({
        "Order_ID": [1, 2, 3],
        "Order_Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Customer_ID": ["C1", "C2", "C3"],
        "Customer_Name": ["A", "B", "C"],
        "Region": ["West", "East", "South"],
        "Category": ["Electronics", "Electronics", "Furniture"],
        "Product": ["P1", "P2", "P3"],
        "Quantity": [2, 3, 1],
        "Unit_Price": [100.0, -50.0, 300.0],
        "Discount": [0.0, 0.1, 1.5]
    })
    cleaned_df, stats = clean_data(invalid_df)
    assert stats["invalid_rows_dropped"] == 2
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["Order_ID"] == 1
