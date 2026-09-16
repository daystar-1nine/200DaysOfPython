
import pytest
import pandas as pd
import numpy as np
from app.data.cleaner import clean_data
from app.preprocessing import prepare_data

@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        'Tenure': [1, 24, 12, 60, 2],
        'MonthlyCharges': [50.0, 60.0, 70.0, 100.0, 45.0],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes']
    })

def test_clean_data(dummy_data):
    df_missing = dummy_data.copy()
    df_missing.loc[0, 'Tenure'] = np.nan
    df_clean = clean_data(df_missing)
    assert not df_clean['Tenure'].isnull().any()

def test_prepare_data(dummy_data):
    X_train, X_test, y_train, y_test, features = prepare_data(dummy_data, test_size=0.4, random_state=42)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert len(features) > 0
