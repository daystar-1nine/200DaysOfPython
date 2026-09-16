
import pytest
import pandas as pd
import numpy as np
from app.loader import load_data
from app.cleaner import clean_data
from app.validator import validate_data
from app.pipeline import get_preprocessor, prepare_data
from app.models import train_baseline, train_gradient_boosting
from app.business_cost import calculate_business_cost

@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        'Tenure': [1, 24, 12, 60, 2],
        'MonthlyCharges': [50.0, 60.0, 70.0, 100.0, 45.0],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes']
    })

def test_clean_data(dummy_data):
    # Add a missing value to test cleaning
    df_missing = dummy_data.copy()
    df_missing.loc[0, 'Tenure'] = np.nan
    df_clean = clean_data(df_missing)
    assert not df_clean['Tenure'].isnull().any()

def test_validate_data(dummy_data):
    assert validate_data(dummy_data) is True
    
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame())

def test_prepare_data(dummy_data):
    X_train, X_test, y_train, y_test, features, preprocessor = prepare_data(dummy_data, test_size=0.4, random_state=42)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert len(features) > 0

def test_calculate_business_cost():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.4, 0.6])
    threshold = 0.5
    fp_cost = 50
    fn_cost = 200
    
    # preds: [1, 0, 0, 1]
    # TP: 1, FP: 1, FN: 1, TN: 1
    cost, tp, fp, fn, tn = calculate_business_cost(y_true, y_prob, threshold, fp_cost, fn_cost)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    assert tn == 1
    assert cost == 250 # 50*1 + 200*1
