import pandas as pd
from app.validator import validate_raw_data, validate_cleaned_data

def test_validate_raw_data_valid(clean_sample_df):
    valid, errors = validate_raw_data(clean_sample_df)
    assert valid is True
    assert len(errors) == 0

def test_validate_raw_data_missing_column(clean_sample_df):
    bad_df = clean_sample_df.drop(columns=['Churn'])
    valid, errors = validate_raw_data(bad_df)
    assert valid is False
    assert any('Churn' in err for err in errors)

def test_validate_raw_data_empty():
    valid, errors = validate_raw_data(pd.DataFrame())
    assert valid is False

def test_validate_cleaned_data_residual_null(clean_sample_df):
    bad_df = clean_sample_df.copy()
    bad_df.loc[0, 'Age'] = None
    valid, errors = validate_cleaned_data(bad_df)
    assert valid is False
    assert any('residual nulls' in err for err in errors)

def test_validate_raw_data_invalid_churn(clean_sample_df):
    bad_df = clean_sample_df.copy()
    bad_df.loc[0, 'Churn'] = 99
    valid, errors = validate_raw_data(bad_df)
    assert valid is False
