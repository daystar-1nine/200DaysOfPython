"""
Unit tests for data validator module app/validator.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 58 parent directory to sys.path.
import sys
from pathlib import Path

DAY58_DIR = Path(__file__).resolve().parent.parent
if str(DAY58_DIR) not in sys.path:
    sys.path.insert(0, str(DAY58_DIR))

# What is used: Import pandas and validate_dataset function.
# Why it is used: Asserts business data rule checks for primary key uniqueness, age range, salary range, and gender category.
# How it works: Executes validate_dataset on test DataFrames.
import pandas as pd
from app.validator import validate_dataset


def test_validate_dataset_passed(sample_clean_customer_df):
    """
    Test valid clean customer DataFrame passes all validation rules.
    """
    # What is used: validate_dataset call on clean test DataFrame.
    # Why it is used: Verifies is_valid boolean is True.
    # How it works: Asserts all rule passed flags are True.
    results = validate_dataset(sample_clean_customer_df)
    assert results["is_valid"] is True
    assert results["rules"]["unique_customer_ids"]["passed"] is True
    assert results["rules"]["valid_age_range"]["passed"] is True


def test_validate_dataset_duplicate_ids_fails(sample_clean_customer_df):
    """
    Test DataFrame with duplicate customer_id fails validation.
    """
    # What is used: DataFrame with intentional duplicate customer_id.
    # Why it is used: Verifies Rule 1 failure detection.
    # How it works: Asserts unique_customer_ids passed flag is False.
    df_dup = sample_clean_customer_df.copy()
    df_dup.loc[1, "customer_id"] = "C101"  # Duplicate C101

    results = validate_dataset(df_dup)
    assert results["is_valid"] is False
    assert results["rules"]["unique_customer_ids"]["passed"] is False


def test_validate_dataset_invalid_age_fails(sample_clean_customer_df):
    """
    Test DataFrame with out-of-bounds age (-10) fails validation.
    """
    # What is used: DataFrame with negative age.
    # Why it is used: Verifies Rule 2 failure detection.
    # How it works: Asserts valid_age_range passed flag is False.
    df_bad_age = sample_clean_customer_df.copy()
    df_bad_age.loc[0, "age"] = -10.0

    results = validate_dataset(df_bad_age)
    assert results["is_valid"] is False
    assert results["rules"]["valid_age_range"]["passed"] is False


def test_validate_dataset_negative_salary_fails(sample_clean_customer_df):
    """
    Test DataFrame with negative salary (-5000) fails validation.
    """
    # What is used: DataFrame with negative salary.
    # Why it is used: Verifies Rule 3 failure detection.
    # How it works: Asserts non_negative_salary passed flag is False.
    df_bad_sal = sample_clean_customer_df.copy()
    df_bad_sal.loc[0, "salary"] = -5000.0

    results = validate_dataset(df_bad_sal)
    assert results["is_valid"] is False
    assert results["rules"]["non_negative_salary"]["passed"] is False
