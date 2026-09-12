import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class AppConfig:
    def __init__(self, data_path, random_state, test_size, cv_folds, target_col):
        self.data_path = data_path
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.target_col = target_col

@pytest.fixture
def config():
    return AppConfig(
        data_path="dummy.csv",
        random_state=42,
        test_size=0.2,
        cv_folds=5,
        target_col="Sales"
    )

@pytest.fixture
def sample_df():
    np.random.seed(42)
    data = {
        "Date": pd.date_range("2023-01-01", periods=50),
        "Region": np.random.choice(["North", "South", "East", "West"], 50),
        "Product_Category": np.random.choice(["Electronics", "Clothing", "Home"], 50),
        "TV_Ad_Spend": np.random.uniform(10, 100, 50),
        "Radio_Ad_Spend": np.random.uniform(5, 50, 50),
        "Social_Media_Ad_Spend": np.random.uniform(1, 20, 50),
        "Discount": np.random.uniform(0, 0.5, 50),
        "Competitor_Price_Ratio": np.random.uniform(0.8, 1.2, 50),
        "Customer_Rating": np.random.uniform(1, 5, 50),
        "Is_Holiday": np.random.choice([0, 1], 50, p=[0.9, 0.1]),
        "Sales": np.random.uniform(100, 500, 50)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_xy():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
    # y = 5 + 2x + 0.5x^2 + epsilon
    y = 5 + 2 * X.ravel() + 0.5 * (X.ravel() ** 2) + np.random.normal(0, 1, 100)
    return X, y
