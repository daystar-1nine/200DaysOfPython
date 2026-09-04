"""
Shared Pytest Fixtures for Day 59 E-Commerce EDA Engine.
"""

# What is used: Import pytest, pandas, and numpy modules.
# Why it is used: Provides reusable test dataset fixtures across unit test modules.
# How it works: Instantiates sample raw sales DataFrame and temporary directory fixtures.
import os
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_sales_df() -> pd.DataFrame:
    """
    Provide raw transactional sales DataFrame with known metrics, missing values, and duplicate row.
    """
    data = {
        "Order_ID": ["ORD101", "ORD102", "ORD103", "ORD104", "ORD105", "ORD101"],
        "Order_Date": ["2026-01-05", "2026-01-15", "2026-02-10", "2026-02-20", "2026-03-05", "2026-01-05"],
        "Customer_ID": ["C101", "C102", "C103", "C101", "C104", "C101"],
        "Customer_Name": [" Rahul Sharma ", "Priya Patel", "Aman Verma", "Rahul Sharma", "Sneha Kulkarni", " Rahul Sharma "],
        "Region": ["North", "South", "East", "West", "North", "North"],
        "Category": ["Electronics", "Fashion", "Electronics", "Home & Kitchen", "Electronics", "Electronics"],
        "Product": ["Laptop Pro", "Running Shoes", "Smartphone X", "Air Fryer", "Laptop Pro", "Laptop Pro"],
        "Quantity": [2, 1, 3, 1, 5, 2],
        "Unit_Price": [60000.0, 3000.0, 40000.0, 700000.0, 60000.0, 60000.0],
        "Discount": [0.10, 0.0, 0.05, 0.15, 0.0, 0.10],
        "Cost_Price": [45000.0, 1800.0, 30000.0, 5000.0, 45000.0, 45000.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_raw_sales_df) -> str:
    """
    Export sample raw sales DataFrame to a temporary CSV file path.
    """
    file_p = tmp_path / "test_sales.csv"
    sample_raw_sales_df.to_csv(file_p, index=False)
    return str(file_p)
