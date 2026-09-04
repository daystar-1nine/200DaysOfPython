"""
Pytest configuration file and reusable test fixtures for Day 57.
"""

# What is used: Import pytest and pandas modules.
# Why it is used: Declares pytest fixture decorators and sample DataFrame construction.
# How it works: Registers fixtures in test environment.
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_sales_df() -> pd.DataFrame:
    """
    Fixture providing a raw sample sales DataFrame with missing values, duplicate Order_IDs, and invalid records.
    """
    # What is used: Dictionary containing mock uncleaned sales data.
    # Why it is used: Simulates raw input dataset for testing loader, cleaner, and transformer modules.
    # How it works: Constructs DataFrame with deliberate duplicate (1001), missing value, and invalid row (-5 Quantity).
    data = {
        "Order_ID": [1001, 1002, 1003, 1001, 1004, 1005],
        "Order_Date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-05", "2026-02-01", "2026-02-05"],
        "Customer_ID": ["C001 ", "C002", "C003", "C001 ", "C004", "C005"],
        "Customer_Name": [" Rahul Sharma", "Priya Patel", "Aman Verma", " Rahul Sharma", "Sneha Kulkarni", "Vikram Singh"],
        "Region": ["West", "East", "South", "West", "North", "West"],
        "Category": ["Electronics", "Electronics", "Furniture", "Electronics", "Appliances", "Furniture"],
        "Product": ["Laptop", "Phone", "Chair", "Laptop", "Refrigerator", "Desk"],
        "Quantity": [2, 3, None, 2, -5, 1],
        "Unit_Price": [60000.0, 30000.0, 4000.0, 60000.0, 45000.0, 12000.0],
        "Discount": [0.05, 0.10, 0.00, 0.05, 0.15, 0.05]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_raw_sales_df) -> str:
    """
    Fixture creating a temporary CSV file populated with raw sales data.
    """
    # What is used: tmp_path fixture provided by pytest.
    # Why it is used: Isolates file reading tests within temporary filesystem directory.
    # How it works: Saves sample DataFrame to CSV and returns file path string.
    file_p = tmp_path / "test_sales.csv"
    sample_raw_sales_df.to_csv(file_p, index=False)
    return str(file_p)


@pytest.fixture
def sample_clean_sales_df() -> pd.DataFrame:
    """
    Fixture providing a cleaned and transformed sales DataFrame.
    """
    # What is used: Dictionary containing valid sales records.
    # Why it is used: Input data for analyzer unit tests.
    # How it works: Provides pre-cleaned sales DataFrames with computed Revenue and Month columns.
    data = {
        "Order_ID": [1001, 1002, 1003, 1004],
        "Order_Date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-02-01"]),
        "Customer_ID": ["C001", "C002", "C003", "C004"],
        "Customer_Name": ["Rahul Sharma", "Priya Patel", "Aman Verma", "Sneha Kulkarni"],
        "Region": ["West", "East", "South", "North"],
        "Category": ["Electronics", "Electronics", "Furniture", "Appliances"],
        "Product": ["Laptop", "Phone", "Chair", "Refrigerator"],
        "Quantity": [2, 3, 5, 1],
        "Unit_Price": [60000.0, 30000.0, 4000.0, 45000.0],
        "Discount": [0.05, 0.10, 0.00, 0.15],
        "Revenue": [114000.0, 81000.0, 20000.0, 38250.0],
        "Month": ["2026-01", "2026-01", "2026-01", "2026-02"],
        "Discount_Range": ["Low (1-9%)", "Medium (10-19%)", "No Discount (0%)", "Medium (10-19%)"]
    }
    return pd.DataFrame(data)
