import pytest
from app.validator import validate_data

def test_validate_passes(sample_df, config):
    assert validate_data(sample_df, config) is True

def test_validate_missing_columns(sample_df, config):
    df_missing = sample_df.drop(columns=["Customer_ID"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df_missing, config)

def test_validate_negative_age(sample_df, config):
    df_neg = sample_df.copy()
    df_neg.loc[0, "Age"] = -1
    with pytest.raises(ValueError, match="negative values in Age"):
        validate_data(df_neg, config)

def test_validate_negative_charges(sample_df, config):
    df_neg = sample_df.copy()
    df_neg.loc[0, "Monthly_Charges"] = -5.0
    with pytest.raises(ValueError, match="negative values in Monthly_Charges"):
        validate_data(df_neg, config)

def test_validate_invalid_churn(sample_df, config):
    df_inv = sample_df.copy()
    df_inv.loc[0, "Churn"] = 2
    with pytest.raises(ValueError, match="Invalid Churn values"):
        validate_data(df_inv, config)

def test_validate_invalid_risk(sample_df, config):
    df_inv = sample_df.copy()
    df_inv.loc[0, "Risk_Level"] = "Extreme"
    with pytest.raises(ValueError, match="Invalid Risk_Level values"):
        validate_data(df_inv, config)
