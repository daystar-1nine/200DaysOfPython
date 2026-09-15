import pandas as pd
from app.data.validator import validate_raw_data, validate_processed_data

def test_validate_raw_valid(sample_clean_df):
    valid, errs = validate_raw_data(sample_clean_df)
    assert valid is True
    assert len(errs) == 0

def test_validate_raw_missing_column(sample_clean_df):
    bad = sample_clean_df.drop(columns=['Churn'])
    valid, errs = validate_raw_data(bad)
    assert valid is False
    assert any('Churn' in e for e in errs)

def test_validate_raw_invalid_churn(sample_clean_df):
    bad = sample_clean_df.copy()
    bad.loc[0, 'Churn'] = 99
    valid, errs = validate_raw_data(bad)
    assert valid is False

def test_validate_processed_residual_null(sample_clean_df):
    bad = sample_clean_df.copy()
    bad.loc[0, 'Age'] = None
    valid, errs = validate_processed_data(bad)
    assert valid is False

def test_validate_empty_dataframe():
    valid, errs = validate_raw_data(pd.DataFrame())
    assert valid is False
