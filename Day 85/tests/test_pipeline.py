
import pytest
import pandas as pd
from app.preprocessing import prep_churn_data, prep_text_data
from app.evaluation import calculate_business_cost
import numpy as np

@pytest.fixture
def dummy_churn_data():
    return pd.DataFrame({
        'Tenure': [1, 24, 12, 60, 2],
        'MonthlyCharges': [50.0, 60.0, 70.0, 100.0, 45.0],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes']
    })

@pytest.fixture
def dummy_text_data():
    return pd.DataFrame({
        'message': ['payment error', 'account issue', 'internet down', 'billing issue', 'slow network'],
        'category': ['Payment', 'Account', 'Technical', 'Billing', 'Technical']
    })

def test_prep_churn(dummy_churn_data):
    X_train, X_test, y_train, y_test, preprocessor = prep_churn_data(dummy_churn_data)
    assert len(X_train) == 4
    assert len(X_test) == 1

def test_prep_text(dummy_text_data):
    X_train, X_test, y_train, y_test = prep_text_data(dummy_text_data)
    assert len(X_train) == 4
    assert len(X_test) == 1

def test_business_cost():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.9, 0.1, 0.1])
    # threshold 0.5 -> pred [1, 1, 0, 0]
    # TP: 1, FP: 1, FN: 1, TN: 1
    # fp_cost = 100, fn_cost = 1000
    cost = calculate_business_cost(y_true, y_prob, 0.5, 100, 1000)
    assert cost == 1100
