"""
Shared Pytest Fixtures for Day 60 BI Analytics Engine.
"""

# What is used: Import pytest, pandas, and numpy modules.
# Why it is used: Provides shared sample datasets for automated unit testing.
# How it works: Instantiates sample sales DataFrames and temporary CSV file paths.
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_sales_df() -> pd.DataFrame:
    """
    Sample raw transaction dataset containing valid rows, whitespace, negative value, and duplicate row.
    """
    data = {
        "Order_ID": ["ORD-101", "ORD-102", "ORD-103", "ORD-104", "ORD-105", "ORD-101"],
        "Order_Date": ["2026-01-10", "2026-01-20", "2026-02-15", "2026-02-25", "2026-03-05", "2026-01-10"],
        "Customer_ID": ["C101", "C102", "C103", "C101", "C104", "C101"],
        "Customer_Name": [" Rahul Sharma ", "Priya Patel", "Aman Verma", "Rahul Sharma", "Sneha Kulkarni", " Rahul Sharma "],
        "Region": ["North", "South", "East", "West", "North", "North"],
        "Category": ["Electronics", "Furniture", "Electronics", "Apparel", "Electronics", "Electronics"],
        "Product": ["MacBook Pro", "Standing Desk", "iPhone 15", "Formal Blazer", "MacBook Pro", "MacBook Pro"],
        "Quantity": [2, 1, 3, 2, 4, 2],
        "Unit_Price": [120000.0, 28000.0, 75000.0, 7500.0, 120000.0, 120000.0],
        "Cost_Price": [95000.0, 18000.0, 58000.0, 4500.0, 95000.0, 95000.0],
        "Discount": [0.10, 0.0, 0.05, 0.15, 0.0, 0.10]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path, sample_raw_sales_df) -> str:
    """
    Write sample raw sales dataset to a temporary file path.
    """
    target = tmp_path / "temp_sales.csv"
    sample_raw_sales_df.to_csv(target, index=False)
    return str(target)
