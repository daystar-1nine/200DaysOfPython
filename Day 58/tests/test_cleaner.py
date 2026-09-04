"""
Unit tests for data cleaner module app/cleaner.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 58 parent directory to sys.path.
import sys
from pathlib import Path

DAY58_DIR = Path(__file__).resolve().parent.parent
if str(DAY58_DIR) not in sys.path:
    sys.path.insert(0, str(DAY58_DIR))

# What is used: Import pandas, numpy, and clean_customer_data function.
# Why it is used: Asserts column normalization, string trimming, gender standardization, currency parsing, age validation, date parsing, deduplication, and imputation.
# How it works: Executes clean_customer_data on test messy DataFrame.
import numpy as np
import pandas as pd
from app.cleaner import clean_customer_data


def test_clean_column_header_normalization(sample_raw_messy_df):
    """
    Test column headers are stripped, lowercased, and underscored.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts column names match expected lower-cased python identifiers.
    # How it works: Checks column names list.
    cleaned_df, _ = clean_customer_data(sample_raw_messy_df)
    assert "customer_id" in cleaned_df.columns
    assert "join_date" in cleaned_df.columns
    assert "Customer_ID" not in cleaned_df.columns


def test_clean_string_trimming_and_casing(sample_raw_messy_df):
    """
    Test leading/trailing whitespace is stripped and Title Case / lowercase applied.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Confirms string values are trimmed and correctly cased.
    # How it works: Asserts name == 'Rahul Sharma', city == 'Mumbai', email == 'rahul@example.com'.
    cleaned_df, _ = clean_customer_data(sample_raw_messy_df)
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C101", "name"].values[0] == "Rahul Sharma"
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C101", "city"].values[0] == "Mumbai"
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C101", "email"].values[0] == "rahul@example.com"


def test_clean_gender_standardization(sample_raw_messy_df):
    """
    Test gender values M, male, MALE, female, F are standardized to Male or Female.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts gender column contains only 'Male', 'Female', or 'Unknown'.
    # How it works: Checks unique values in gender column.
    cleaned_df, _ = clean_customer_data(sample_raw_messy_df)
    genders = set(cleaned_df["gender"].unique())
    assert genders.issubset({"Male", "Female", "Unknown"})
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C101", "gender"].values[0] == "Male"
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C102", "gender"].values[0] == "Female"


def test_clean_salary_currency_parsing(sample_raw_messy_df):
    """
    Test currency strings ₹60,000 and unknown are parsed into numeric floats.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts salary column is float dtype and contains valid numbers.
    # How it works: Asserts C101 salary is 60000.0 and C103 (unknown) is imputed.
    cleaned_df, stats = clean_customer_data(sample_raw_messy_df)
    assert pd.api.types.is_float_dtype(cleaned_df["salary"])
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C101", "salary"].values[0] == 60000.0
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C103", "salary"].values[0] > 0


def test_clean_age_coercion_and_range_validation(sample_raw_messy_df):
    """
    Test invalid ages (-5) are detected and set to median.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts invalid_ages_corrected statistic is > 0 and clean ages are between 0 and 120.
    # How it works: Checks stats dictionary and age range.
    cleaned_df, stats = clean_customer_data(sample_raw_messy_df)
    assert stats["invalid_ages_corrected"] == 1
    assert cleaned_df["age"].between(0, 120).all()


def test_clean_join_date_datetime_parsing(sample_raw_messy_df):
    """
    Test join_date column is converted to datetime and invalid dates are handled.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts join_date is datetime64 dtype.
    # How it works: Checks dtype of join_date column.
    cleaned_df, stats = clean_customer_data(sample_raw_messy_df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["join_date"])
    assert stats["invalid_dates_corrected"] == 1


def test_clean_duplicate_customer_ids_removed(sample_raw_messy_df):
    """
    Test duplicate customer_id C101 is removed.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts duplicates_removed stat is 1 and customer_id is unique.
    # How it works: Checks deduplication stats and customer_id uniqueness.
    cleaned_df, stats = clean_customer_data(sample_raw_messy_df)
    assert stats["duplicates_removed"] == 1
    assert len(cleaned_df) == 4
    assert not cleaned_df["customer_id"].duplicated().any()


def test_clean_missing_value_imputation(sample_raw_messy_df):
    """
    Test missing Age, Salary, City, Department, Email, Phone are imputed.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Asserts 0 remaining NaNs in cleaned DataFrame.
    # How it works: Checks null count across columns.
    cleaned_df, stats = clean_customer_data(sample_raw_messy_df)
    assert stats["nulls_filled"] > 0
    assert cleaned_df.isna().sum().sum() == 0


def test_clean_derived_date_features(sample_raw_messy_df):
    """
    Test join_year, join_month, and join_month_name derived features are added.
    """
    # What is used: clean_customer_data execution.
    # Why it is used: Verifies derived date columns exist and are populated.
    # How it works: Checks join_year, join_month, join_month_name columns.
    cleaned_df, _ = clean_customer_data(sample_raw_messy_df)
    assert "join_year" in cleaned_df.columns
    assert "join_month" in cleaned_df.columns
    assert "join_month_name" in cleaned_df.columns
    assert cleaned_df.loc[0, "join_year"] == 2026
    assert cleaned_df.loc[0, "join_month_name"] == "January"
