import pytest
import pandas as pd

class FeatureEngineer:
    def engineer(self, df):
        df = df.copy()
        df['Total_Ad_Spend'] = df['TV_Ad_Spend'] + df['Radio_Ad_Spend'] + df['Social_Media_Ad_Spend']
        df['Ad_Efficiency'] = df['Sales'] / (df['Total_Ad_Spend'] + 1e-5)
        df['Discount_Intensity'] = df['Discount'] * df['Competitor_Price_Ratio']
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
        return df

def test_engineer_adds_total_ad_spend(sample_df):
    engineer = FeatureEngineer()
    df = engineer.engineer(sample_df)
    assert 'Total_Ad_Spend' in df.columns

def test_engineer_total_ad_spend_value(sample_df):
    engineer = FeatureEngineer()
    df = engineer.engineer(sample_df)
    expected = sample_df['TV_Ad_Spend'] + sample_df['Radio_Ad_Spend'] + sample_df['Social_Media_Ad_Spend']
    pd.testing.assert_series_equal(df['Total_Ad_Spend'], expected, check_names=False)

def test_engineer_adds_ad_efficiency(sample_df):
    engineer = FeatureEngineer()
    df = engineer.engineer(sample_df)
    assert 'Ad_Efficiency' in df.columns

def test_engineer_adds_discount_intensity(sample_df):
    engineer = FeatureEngineer()
    df = engineer.engineer(sample_df)
    assert 'Discount_Intensity' in df.columns

def test_engineer_month_year_present(sample_df):
    engineer = FeatureEngineer()
    df = engineer.engineer(sample_df)
    assert 'Month' in df.columns
    assert 'Year' in df.columns
