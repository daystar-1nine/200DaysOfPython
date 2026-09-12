import pytest
import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()
    if 'contract_type' in df.columns:
        df['contract_type'] = df['contract_type'].str.lower()
    if 'monthly_charges' in df.columns:
        df = df[df['monthly_charges'] >= 0]
    if 'total_charges' in df.columns:
        df['total_charges'] = df['total_charges'].fillna(df['monthly_charges'])
    df = df.drop_duplicates()
    if 'churn' in df.columns:
        df = df[df['churn'].isin([0, 1])]
    return df

def test_clean_standardizes_contract_type(sample_df):
    sample_df['contract_type'] = 'Month-To-Month'
    cleaned = clean_data(sample_df)
    assert cleaned['contract_type'].iloc[0] == 'month-to-month'

def test_clean_removes_negative_values(sample_df):
    sample_df.loc[0, 'monthly_charges'] = -50
    cleaned = clean_data(sample_df)
    assert (cleaned['monthly_charges'] >= 0).all()

def test_clean_imputes_missing_values(sample_df):
    sample_df.loc[0, 'total_charges'] = np.nan
    cleaned = clean_data(sample_df)
    assert not cleaned['total_charges'].isna().any()

def test_clean_removes_duplicates(sample_df):
    df_dup = pd.concat([sample_df, sample_df.iloc[[0]]])
    cleaned = clean_data(df_dup)
    assert len(cleaned) == len(sample_df)

def test_clean_valid_churn_targets(sample_df):
    sample_df.loc[0, 'churn'] = 2
    cleaned = clean_data(sample_df)
    assert cleaned['churn'].isin([0, 1]).all()
