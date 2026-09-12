"""
Pytest configuration and shared fixtures for Day 73 test suite.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

DAY73_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DAY73_DIR)
sys.path.insert(0, os.path.join(DAY73_DIR, "app"))

@pytest.fixture
def synthetic_sales_df():
    np.random.seed(42)
    x = np.linspace(10000, 100000, 100)
    noise = np.random.normal(0, 5000, 100)
    y = 20000 + 1.8 * x + noise
    return pd.DataFrame({
        "Advertising_Spend": x,
        "Sales": y,
        "Region": ["North"] * 50 + ["South"] * 50,
        "Month": ["Jan"] * 100
    })

@pytest.fixture
def clean_linear_arrays():
    X = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = np.array([30.0, 50.0, 70.0, 90.0, 110.0])
    return X, y
