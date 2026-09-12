import pytest
import pandas as pd

class DataValidator:
    def __init__(self, config):
        self.config = config
    def validate(self, df):
        if len(df) < 10:
            raise ValueError("Too few rows")
        if (df[self.config.target_col] < 0).any():
            raise ValueError("Negative sales")
        if 'Discount' in df.columns and (df['Discount'] < 0).any() or (df['Discount'] > 1).any():
            raise ValueError("Invalid discount")
        return True

def test_validator_passes_clean_data(sample_df, config):
    validator = DataValidator(config)
    assert validator.validate(sample_df) is True

def test_validator_detects_negative_sales(sample_df, config):
    df = sample_df.copy()
    df.loc[0, config.target_col] = -5
    validator = DataValidator(config)
    with pytest.raises(ValueError):
        validator.validate(df)

def test_validator_detects_invalid_discount(sample_df, config):
    df = sample_df.copy()
    df.loc[0, 'Discount'] = 1.1
    validator = DataValidator(config)
    with pytest.raises(ValueError):
        validator.validate(df)

def test_validator_raises_on_min_rows(config):
    df = pd.DataFrame({'Sales': [1, 2]})
    validator = DataValidator(config)
    with pytest.raises(ValueError):
        validator.validate(df)
