import pytest
import pandas as pd
import numpy as np

try:
    from app.multicollinearity import calculate_vif, flag_high_vif
except ImportError:
    from multicollinearity import calculate_vif, flag_high_vif

@pytest.fixture
def features_df():
    np.random.seed(42)
    X1 = np.random.rand(100)
    X2 = X1 * 0.99 + np.random.randn(100) * 0.01  # Highly collinear with X1
    X3 = np.random.rand(100)
    return pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

def test_vif_returns_dataframe(features_df):
    vif_df = calculate_vif(features_df)
    assert isinstance(vif_df, pd.DataFrame)

def test_vif_has_correct_columns(features_df):
    vif_df = calculate_vif(features_df)
    assert 'Feature' in vif_df.columns
    assert 'VIF' in vif_df.columns

def test_vif_all_positive(features_df):
    vif_df = calculate_vif(features_df)
    assert (vif_df['VIF'] >= 0).all()

def test_flag_high_vif_detects_collinear_features(features_df):
    flags = flag_high_vif(features_df, threshold=10)
    assert len(flags) > 0
    assert any('X1' in f or 'X2' in f for f in flags)
