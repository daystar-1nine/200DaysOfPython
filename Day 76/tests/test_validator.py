import pytest
import pandas as pd

class ValidationError(Exception):
    pass

def validate_data(df):
    if (df.get('age', pd.Series([1])) < 0).any():
        raise ValidationError("Negative age")
    if (df.get('monthly_charges', pd.Series([1])) < 0).any():
        raise ValidationError("Negative charges")
    if not df.get('churn', pd.Series([0, 1])).isin([0, 1]).all():
        raise ValidationError("Invalid churn")
    return True

def test_validator_passes_clean_data(sample_df):
    assert validate_data(sample_df)

def test_validator_detects_negative_age(sample_df):
    sample_df.loc[0, 'age'] = -5
    with pytest.raises(ValidationError, match="Negative age"):
        validate_data(sample_df)

def test_validator_detects_negative_charges(sample_df):
    sample_df.loc[0, 'monthly_charges'] = -10
    with pytest.raises(ValidationError, match="Negative charges"):
        validate_data(sample_df)

def test_validator_detects_invalid_churn(sample_df):
    sample_df.loc[0, 'churn'] = 3
    with pytest.raises(ValidationError, match="Invalid churn"):
        validate_data(sample_df)
