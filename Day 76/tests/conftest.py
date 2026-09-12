import pytest
import pandas as pd
import numpy as np

class AppConfig:
    pass

@pytest.fixture
def config():
    return AppConfig()

@pytest.fixture
def sample_df():
    data = {
        'customer_id': range(60),
        'age': np.random.randint(18, 80, 60),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 60),
        'monthly_charges': np.random.uniform(20, 120, 60),
        'total_charges': np.random.uniform(100, 8000, 60),
        'support_calls': np.random.randint(0, 10, 60),
        'churn': np.random.choice([0, 1], 60)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_classification_data():
    X = pd.DataFrame({'feature1': np.random.randn(60), 'feature2': np.random.randn(60)})
    y = pd.Series(np.random.choice([0, 1], 60))
    probabilities = np.random.uniform(0, 1, 60)
    return X, y, probabilities
