import pandas as pd
import numpy as np
from app.cleaner import clean_dataset

def test_clean_removes_duplicates(raw_sample_df):
    initial_len = len(raw_sample_df)
    df_clean = clean_dataset(raw_sample_df)
    assert len(df_clean) < initial_len

def test_clean_imputes_numeric_nulls(raw_sample_df):
    df_clean = clean_dataset(raw_sample_df)
    assert df_clean['Age'].isnull().sum() == 0
    assert df_clean['Monthly_Charges'].isnull().sum() == 0

def test_clean_imputes_categorical_nulls(raw_sample_df):
    df_clean = clean_dataset(raw_sample_df)
    assert df_clean['Contract_Type'].isnull().sum() == 0

def test_clean_ensures_churn_binary(raw_sample_df):
    df_clean = clean_dataset(raw_sample_df)
    assert set(df_clean['Churn'].unique()).issubset({0, 1})
    assert df_clean['Churn'].dtype in [np.int32, np.int64, int]

def test_clean_idempotency(clean_sample_df):
    df_clean_again = clean_dataset(clean_sample_df)
    assert len(df_clean_again) == len(clean_sample_df)
