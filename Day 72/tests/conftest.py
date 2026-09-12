"""
Pytest configuration and shared fixtures for Day 72 test suite.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add app directory to sys.path
DAY72_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DAY72_DIR)
sys.path.insert(0, os.path.join(DAY72_DIR, "app"))

@pytest.fixture
def sample_numeric_df():
    np.random.seed(42)
    n = 100
    x1 = np.linspace(1, 100, n)
    x2 = 2.0 * x1 + np.random.normal(0, 5, n)
    x3 = 50.0 - 0.5 * x1 + np.random.normal(0, 3, n)
    x4 = np.random.normal(10, 2, n)
    return pd.DataFrame({
        "Feature_A": x1,
        "Feature_B": x2,
        "Feature_C": x3,
        "Feature_D": x4
    })

@pytest.fixture
def clean_linear_arrays():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    return x, y
