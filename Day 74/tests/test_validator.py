import pytest
import pandas as pd

try:
    from app.validator import validate_data
except ImportError:
    from validator import validate_data

def test_validator_passes_clean_data(sample_df):
    assert validate_data(sample_df) is True

def test_validator_detects_negative_sales(sample_df):
    sample_df.loc[0, 'Sales'] = -100
    with pytest.raises(ValueError, match="negative"):
        validate_data(sample_df)

def test_validator_detects_invalid_discount(sample_df):
    sample_df.loc[0, 'Discount'] = 110
    with pytest.raises(ValueError, match="discount"):
        validate_data(sample_df)

def test_validator_raises_on_too_few_rows(sample_df):
    small_df = sample_df.head(5)
    with pytest.raises(ValueError, match="rows"):
        validate_data(small_df, min_rows=10)
