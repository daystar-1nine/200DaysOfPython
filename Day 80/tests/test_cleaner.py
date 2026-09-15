import pandas as pd
import numpy as np
from app.data.cleaner import clean_dataset, generate_data_quality_report

def test_clean_removes_duplicates(sample_raw_df):
    initial_len = len(sample_raw_df)
    clean_df = clean_dataset(sample_raw_df)
    assert len(clean_df) < initial_len

def test_clean_imputes_numerics(sample_raw_df):
    clean_df = clean_dataset(sample_raw_df)
    assert clean_df['Age'].isnull().sum() == 0
    assert clean_df['Monthly_Charges'].isnull().sum() == 0

def test_clean_imputes_categoricals(sample_raw_df):
    clean_df = clean_dataset(sample_raw_df)
    assert clean_df['Contract_Type'].isnull().sum() == 0

def test_clean_target_is_int(sample_raw_df):
    clean_df = clean_dataset(sample_raw_df)
    assert clean_df['Churn'].dtype in [int, np.int32, np.int64]

def test_generate_data_quality_report(sample_raw_df):
    report = generate_data_quality_report(sample_raw_df)
    assert isinstance(report, pd.DataFrame)
    assert 'Column' in report.columns
    assert 'Missing_Count' in report.columns
    assert 'Unique_Values' in report.columns
    assert len(report) == len(sample_raw_df.columns)
