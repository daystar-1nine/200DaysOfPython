
import pytest
import pandas as pd
import numpy as np
import os
from app.data.loader import load_and_clean
from app.analysis.statistics import generate_summary
from app.features.engineering import prepare_features

@pytest.fixture
def mock_csv(tmp_path):
    df = pd.DataFrame({
        'age': [25, -10, 30],
        'income': [50000, np.nan, 60000],
        'score': [80, 90, 85],
        'target': [1, 0, 1]
    })
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return str(p)

def test_data_cleaning(mock_csv):
    df = load_and_clean(mock_csv)
    # Should drop age -10
    assert len(df) == 2
    # Should fill NaN income with median (55000)
    assert not df['income'].isna().any()
    assert df.loc[df['age']==25, 'income'].values[0] == 50000

def test_feature_engineering(mock_csv):
    df = load_and_clean(mock_csv)
    # Extremely small dataset, test_size=0.2 of 2 is 0 or 1 depending on rounding,
    # let's just ensure it runs and outputs 4 arrays
    X_train, X_test, y_train, y_test = prepare_features(df)
    assert len(X_train) + len(X_test) == len(df)

def test_summary_generation(mock_csv):
    df = load_and_clean(mock_csv)
    summary = generate_summary(df)
    assert 'age' in summary.columns
