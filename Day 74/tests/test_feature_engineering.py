import pytest
import pandas as pd

try:
    from app.feature_engineering import engineer_features
except ImportError:
    from feature_engineering import engineer_features

def test_engineer_adds_total_ad_spend(sample_df):
    df = engineer_features(sample_df)
    assert 'Total_Ad_Spend' in df.columns

def test_engineer_total_ad_spend_correct_value(sample_df):
    df = engineer_features(sample_df)
    expected = df['TV_Spend'] + df['Digital_Spend'] + df['Radio_Spend']
    pd.testing.assert_series_equal(df['Total_Ad_Spend'], expected, check_names=False)

def test_engineer_adds_ad_efficiency(sample_df):
    df = engineer_features(sample_df)
    assert 'Ad_Efficiency' in df.columns

def test_engineer_adds_discount_intensity(sample_df):
    df = engineer_features(sample_df)
    assert 'Discount_Intensity' in df.columns

def test_engineer_month_year_present(sample_df):
    df = engineer_features(sample_df)
    assert 'Month' in df.columns
    assert 'Year' in df.columns
