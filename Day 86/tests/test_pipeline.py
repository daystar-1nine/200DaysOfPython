
import pytest
import pandas as pd
from app.preprocessing import prep_data
from sklearn.svm import SVC
import numpy as np

@pytest.fixture
def dummy_churn_data():
    return pd.DataFrame({
        'Tenure': [1, 24, 12, 60, 2, 45, 2, 10, 50, 5],
        'MonthlyCharges': [50.0, 60.0, 70.0, 100.0, 45.0, 80.0, 20.0, 30.0, 90.0, 55.0],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month'],
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
    })

def test_prep_data(dummy_churn_data):
    (X_train, X_test, y_train, y_test), preprocessor = prep_data(dummy_churn_data)
    assert len(X_train) == 8
    assert len(X_test) == 2

def test_svm_kernel():
    clf = SVC(kernel='linear')
    assert clf.kernel == 'linear'
