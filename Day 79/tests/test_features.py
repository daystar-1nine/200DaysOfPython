import numpy as np
import pandas as pd
from app.feature_engineering import engineer_features

def test_engineer_features_adds_columns(clean_sample_df):
    df_feat = engineer_features(clean_sample_df)
    expected = ['Tenure_to_Age_Ratio', 'Charges_Deviation', 'Support_Per_Month', 'High_Risk_Flag']
    for col in expected:
        assert col in df_feat.columns

def test_tenure_to_age_ratio(clean_sample_df):
    df_feat = engineer_features(clean_sample_df)
    expected_ratio = clean_sample_df['Tenure_Months'] / (clean_sample_df['Age'] + 1e-5)
    np.testing.assert_allclose(df_feat['Tenure_to_Age_Ratio'], expected_ratio, rtol=1e-4)

def test_high_risk_flag_binary(clean_sample_df):
    df_feat = engineer_features(clean_sample_df)
    assert set(df_feat['High_Risk_Flag'].unique()).issubset({0, 1})

def test_no_nan_introduced(clean_sample_df):
    df_feat = engineer_features(clean_sample_df)
    assert df_feat.isnull().sum().sum() == 0

def test_support_per_month_positive(clean_sample_df):
    df_feat = engineer_features(clean_sample_df)
    assert (df_feat['Support_Per_Month'] >= 0).all()
