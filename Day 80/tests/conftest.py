import os
import sys
import pytest
import numpy as np
import pandas as pd

DAY80_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY80_DIR not in sys.path:
    sys.path.insert(0, DAY80_DIR)

from app.config import AppConfig
from app.data.cleaner import clean_dataset
from app.preprocessing.pipeline import build_preprocessor, split_dataset

@pytest.fixture
def sample_config():
    return AppConfig()

@pytest.fixture
def sample_raw_df():
    np.random.seed(42)
    n = 120
    data = {
        'Customer_ID': [f'CUST_{i}' for i in range(n)],
        'Age': np.random.randint(18, 75, size=n).astype(float),
        'Gender': np.random.choice(['Male', 'Female'], size=n),
        'Tenure_Months': np.random.randint(1, 60, size=n).astype(float),
        'Monthly_Charges': np.random.uniform(20, 120, size=n),
        'Total_Charges': np.random.uniform(100, 5000, size=n),
        'Contract_Type': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n),
        'Internet_Service': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n),
        'Payment_Method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n),
        'Support_Calls': np.random.randint(0, 7, size=n),
        'Late_Payments': np.random.randint(0, 5, size=n),
        'Usage_Hours': np.random.uniform(10, 200, size=n),
        'Discount': np.random.choice([0.0, 10.0, 20.0], size=n),
        'Complaints': np.random.randint(0, 4, size=n),
        'Churn': np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)
    df.loc[1, 'Age'] = np.nan
    df.loc[2, 'Monthly_Charges'] = np.nan
    df.loc[3, 'Contract_Type'] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df

@pytest.fixture
def sample_clean_df(sample_raw_df):
    return clean_dataset(sample_raw_df)

@pytest.fixture
def sample_splits(sample_clean_df, sample_config):
    feature_cols = sample_config.numeric_features + sample_config.categorical_features
    return split_dataset(sample_clean_df, feature_cols, sample_config.target_column, test_size=0.25, random_state=42)
