"""
Module: validator.py
Validates processed dataset against business data quality rules and domain boundaries.
"""

# What is used: Import pandas library.
# Why it is used: Core package for data validation rules execution.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate cleaned customer DataFrame against business rules.

    Validation Rules:
    1. Primary Key: customer_id is present and has 0 duplicates.
    2. Age Domain: all ages reside within 0 to 120.
    3. Salary Domain: all salaries are non-negative (>= 0).
    4. Gender Domain: all gender values are in ['Male', 'Female', 'Unknown'].
    5. Completeness: zero missing values in primary columns.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        dict: Validation audit dictionary with pass/fail boolean flags and detailed rule metrics.
    """
    rules = {}

    # Rule 1: Customer ID Uniqueness
    if "customer_id" in df.columns:
        dup_ids = int(df["customer_id"].duplicated().sum())
        rules["unique_customer_ids"] = {
            "passed": dup_ids == 0,
            "duplicate_count": dup_ids
        }

    # Rule 2: Age Domain (0-120)
    if "age" in df.columns:
        invalid_ages = int((~df["age"].between(0, 120)).sum())
        rules["valid_age_range"] = {
            "passed": invalid_ages == 0,
            "invalid_count": invalid_ages
        }

    # Rule 3: Salary Domain (>= 0)
    if "salary" in df.columns:
        neg_salaries = int((df["salary"] < 0).sum())
        rules["non_negative_salary"] = {
            "passed": neg_salaries == 0,
            "invalid_count": neg_salaries
        }

    # Rule 4: Gender Standard Categorization
    if "gender" in df.columns:
        allowed = {"Male", "Female", "Unknown"}
        unmapped_genders = int((~df["gender"].isin(allowed)).sum())
        rules["standard_gender_category"] = {
            "passed": unmapped_genders == 0,
            "invalid_count": unmapped_genders
        }

    # Rule 5: Zero Remaining NaNs in Critical Columns
    crit_cols = [col for col in ["customer_id", "name", "age", "salary", "gender"] if col in df.columns]
    null_count = int(df[crit_cols].isna().sum().sum())
    rules["zero_missing_critical"] = {
        "passed": null_count == 0,
        "missing_count": null_count
    }

    # Overall Validation Status
    all_passed = all(r["passed"] for r in rules.values())

    return {
        "is_valid": all_passed,
        "rules": rules
    }
