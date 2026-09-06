"""
Pytest fixtures for Day 65 test suite.
"""

import os
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_odd_data():
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

@pytest.fixture
def sample_even_data():
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

@pytest.fixture
def sample_outlier_data():
    # Normal cluster around 20, one extreme 200, one negative -50
    return pd.Series([18.0, 19.0, 20.0, 21.0, 22.0, 200.0, -50.0])

@pytest.fixture
def constant_data():
    return pd.Series([42.0, 42.0, 42.0, 42.0])

@pytest.fixture
def nan_inf_data():
    return pd.Series([10.0, 20.0, np.nan, 30.0, np.inf, -np.inf, 40.0])

@pytest.fixture
def empty_data():
    return pd.Series([], dtype=float)

@pytest.fixture
def real_sales_df():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ecommerce_sales.csv")
    return pd.read_csv(data_path)
