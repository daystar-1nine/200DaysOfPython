"""
Pytest Fixtures and Mock Data Configuration for Day 61.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_sales_df():
    """Provides a deterministic mock DataFrame matching the sales schema."""
    dates = pd.date_range("2026-01-01", periods=12, freq="MS")
    data = {
        "Order_ID": [f"ORD-{i:03d}" for i in range(1, 13)],
        "Order_Date": dates,
        "Year_Month": dates.strftime("%Y-%m"),
        "Customer_ID": [f"C{100 + (i % 3)}" for i in range(1, 13)],
        "Customer_Name": ["Customer Alpha", "Customer Beta", "Customer Gamma"] * 4,
        "Region": ["North", "South", "East", "West"] * 3,
        "Category": ["Electronics", "Furniture", "Apparel"] * 4,
        "Product": ["Laptop Pro", "Ergonomic Chair", "Running Shoes"] * 4,
        "Quantity": [1, 2, 3, 4, 2, 1, 3, 5, 2, 4, 1, 2],
        "Unit_Price": [50000, 15000, 4000] * 4,
        "Cost_Price": [35000, 10000, 2500] * 4,
        "Discount": [0.05, 0.10, 0.0] * 4,
        "Revenue": [47500.0, 27000.0, 12000.0, 190000.0, 27000.0, 4000.0, 142500.0, 67500.0, 8000.0, 190000.0, 14250.0, 8000.0],
        "Cost": [35000, 20000, 7500, 140000, 20000, 2500, 105000, 50000, 5000, 140000, 10000, 5000],
        "Profit": [12500.0, 7000.0, 4500.0, 50000.0, 7000.0, 1500.0, 37500.0, 17500.0, 3000.0, 50000.0, 4250.0, 3000.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_path(tmp_path, sample_sales_df):
    """Writes mock data to a temporary CSV file."""
    csv_file = tmp_path / "mock_sales.csv"
    sample_sales_df.to_csv(csv_file, index=False)
    return csv_file
