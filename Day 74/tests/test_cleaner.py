import pytest
import pandas as pd
import numpy as np

try:
    from app.cleaner import clean_data
except ImportError:
    from cleaner import clean_data

def test_clean_removes_null_sales(sample_df):
    sample_df.loc[0, 'Sales'] = np.nan
    cleaned = clean_data(sample_df)
    assert not cleaned['Sales'].isnull().any()
    assert len(cleaned) == len(sample_df) - 1

def test_clean_removes_negative_sales(sample_df):
    sample_df.loc[1, 'Sales'] = -500
    cleaned = clean_data(sample_df)
    assert (cleaned['Sales'] >= 0).all()

def test_clean_removes_duplicates(sample_df):
    df_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(df_dup)
    assert len(cleaned) == len(sample_df)

def test_clean_clips_discount_to_100(sample_df):
    sample_df.loc[2, 'Discount'] = 150
    cleaned = clean_data(sample_df)
    assert cleaned['Discount'].max() <= 100

def test_clean_removes_unknown_region(sample_df):
    sample_df.loc[3, 'Region'] = 'Unknown'
    cleaned = clean_data(sample_df)
    assert 'Unknown' not in cleaned['Region'].values
