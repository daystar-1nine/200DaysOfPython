"""
Pytest Configuration and Mock Fixtures for Day 62.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def mock_ecommerce_df():
    """Generates a synthetic e-commerce DataFrame for deterministic unit testing."""
    dates = pd.date_range("2026-01-01", periods=12, freq="MS")
    data = {
        "Order_ID": [f"ORD-{i:03d}" for i in range(1, 13)],
        "Order_Date": dates,
        "Year_Month": dates.strftime("%Y-%m"),
        "Customer_ID": [f"C{100 + (i % 4)}" for i in range(1, 13)],
        "Customer_Name": ["Customer Alpha", "Customer Beta", "Customer Gamma", "Customer Delta"] * 3,
        "Region": ["North", "South", "East", "West"] * 3,
        "Category": ["Electronics", "Furniture", "Apparel"] * 4,
        "Product": ["Pro Laptop", "Ergo Chair", "Running Shoes"] * 4,
        "Quantity": [1, 2, 3, 4, 1, 2, 3, 2, 1, 4, 2, 3],
        "Unit_Price": [60000, 18000, 4500] * 4,
        "Cost_Price": [42000, 12000, 2800] * 4,
        "Discount": [0.05, 0.10, 0.0] * 4,
        "Revenue": [57000.0, 32400.0, 13500.0, 228000.0, 32400.0, 9000.0, 171000.0, 64800.0, 4500.0, 228000.0, 32400.0, 13500.0],
        "Cost": [42000, 24000, 8400, 168000, 24000, 5600, 126000, 48000, 2800, 168000, 24000, 8400],
        "Profit": [15000.0, 8400.0, 5100.0, 60000.0, 8400.0, 3400.0, 45000.0, 16800.0, 1700.0, 60000.0, 8400.0, 5100.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_path(tmp_path, mock_ecommerce_df):
    """Writes mock data to a temporary CSV."""
    fpath = tmp_path / "mock_ecommerce.csv"
    mock_ecommerce_df.to_csv(fpath, index=False)
    return fpath