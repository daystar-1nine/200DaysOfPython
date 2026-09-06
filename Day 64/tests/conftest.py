"""
Pytest Test Fixtures for Day 64 Advanced Statistical EDA Engine
==============================================================
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_sales_df():
    """
    Synthetic dataset mirroring production ecommerce_sales.csv schema.
    """
    np.random.seed(42)
    n = 60
    categories = ["Electronics", "Furniture", "Fitness", "Kitchenware", "Apparel"]
    regions = ["North", "South", "East", "West"]

    df = pd.DataFrame({
        "Order_ID": [f"ORD-{2000 + i}" for i in range(n)],
        "Order_Date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Customer_ID": [f"C{100 + (i % 12)}" for i in range(n)],
        "Customer_Name": [f"Client_{i % 12}" for i in range(n)],
        "Region": [regions[i % len(regions)] for i in range(n)],
        "Category": [categories[i % len(categories)] for i in range(n)],
        "Product": [f"Product_{i % 8}" for i in range(n)],
        "Quantity": np.random.randint(1, 10, size=n),
        "Unit_Price": np.random.randint(2000, 60000, size=n),
        "Cost_Price": np.random.randint(1500, 45000, size=n),
        "Discount": np.random.choice([0.05, 0.10, 0.15, 0.20], size=n),
        "Revenue": np.random.uniform(5000, 180000, size=n),
        "Cost": np.random.uniform(3500, 140000, size=n),
        "Profit": np.random.uniform(1000, 40000, size=n),
        "Year": [2026] * n,
        "Month": [(i % 6) + 1 for i in range(n)],
        "Month_Name": ["January"] * n
    })
    return df

@pytest.fixture
def clean_sample_df(sample_sales_df):
    """
    Returns pre-cleaned sample data.
    """
    from app.cleaner import clean_sales_data
    return clean_sales_data(sample_sales_df)
