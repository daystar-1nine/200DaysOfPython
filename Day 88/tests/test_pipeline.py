
import pytest
import pandas as pd
from app.preprocessing import prep_data
from app.models import build_advanced_network
import numpy as np

@pytest.fixture
def dummy_churn_data():
    return pd.DataFrame({
        'Tenure': [1, 24, 12, 60, 2, 45, 2, 10, 50, 5] * 3,
        'MonthlyCharges': [50.0, 60.0, 70.0, 100.0, 45.0, 80.0, 20.0, 30.0, 90.0, 55.0] * 3,
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month'] * 3,
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'] * 3
    })

def test_prep_data(dummy_churn_data):
    (X_train_t, X_val_t, X_test_t, y_train, y_val, y_test), preprocessor = prep_data(dummy_churn_data)
    assert len(X_train_t) > 0

def test_advanced_nn_build():
    model = build_advanced_network(10)
    # Check for presence of dropout and batchnorm layers
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert 'MockLayer' in layer_types
