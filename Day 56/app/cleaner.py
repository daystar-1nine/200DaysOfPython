"""
Module: cleaner.py
Performs data cleaning, duplicate removal, NaN imputation, and score validation pipelines.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Essential for DataFrame cleaning, data type conversion, and imputation.
# How it works: Brings pandas and numpy namespaces into scope.
import numpy as np
import pandas as pd


def clean_student_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean student DataFrame by dropping duplicates, imputing missing marks,
    validating score ranges (0-100), and stripping string whitespace.

    Args:
        df: Raw student DataFrame.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and dictionary of cleaning audit stats.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures cleaning mutations do not affect raw input buffer.
    # How it works: Creates independent DataFrame object in memory.
    cleaned_df = df.copy()
    stats = {
        "initial_rows": len(cleaned_df),
        "duplicates_removed": 0,
        "nulls_filled": 0,
        "invalid_scores_dropped": 0,
        "final_rows": 0
    }

    # What is used: String stripping with .str.strip().
    # Why it is used: Cleans accidental leading/trailing whitespace in string columns.
    # How it works: Applies Python str.strip() across all text Series elements.
    for col in ["Student_ID", "Name", "Department"]:
        if col in cleaned_df.columns and pd.api.types.is_string_dtype(cleaned_df[col]):
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # What is used: df.drop_duplicates(subset=["Student_ID"], keep="first").
    # Why it is used: Eliminates duplicate student records using Student_ID primary key.
    # How it works: Identifies repeated Student_ID values and retains only the first occurrence.
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["Student_ID"], keep="first")
    stats["duplicates_removed"] = initial_count - len(cleaned_df)

    # What is used: pd.to_numeric() with errors='coerce'.
    # Why it is used: Converts mark columns to numeric float types, replacing unparseable values with NaN.
    # How it works: Coerces invalid string numbers to np.nan floats.
    subject_cols = ["Math", "Physics", "Chemistry"]
    for col in subject_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    # What is used: Null counting with .isnull().sum() and median imputation with .fillna().
    # Why it is used: Safely imputes missing numerical marks using the median score of that subject.
    # How it works: Calculates subject median score ignoring NaNs, then replaces NaNs with median value.
    total_nulls = int(cleaned_df[subject_cols].isnull().sum().sum())
    stats["nulls_filled"] = total_nulls

    for col in subject_cols:
        median_val = cleaned_df[col].median()
        # Fallback if median is NaN (e.g. all empty)
        if pd.isna(median_val):
            median_val = 0.0
        cleaned_df[col] = cleaned_df[col].fillna(round(float(median_val), 1))

    # What is used: Boolean range condition (col >= 0) & (col <= 100).
    # Why it is used: Validates that all scores reside within the legitimate 0-100 academic boundary.
    # How it works: Filters out rows containing out-of-bounds scores.
    valid_mask = pd.Series(True, index=cleaned_df.index)
    for col in subject_cols:
        valid_mask &= (cleaned_df[col] >= 0.0) & (cleaned_df[col] <= 100.0)

    pre_filter_count = len(cleaned_df)
    cleaned_df = cleaned_df[valid_mask].copy()
    stats["invalid_scores_dropped"] = pre_filter_count - len(cleaned_df)

    # What is used: reset_index(drop=True).
    # Why it is used: Re-indexes cleaned DataFrame continuously from 0 to N-1.
    # How it works: Discards old sparse index labels and generates fresh sequential integer range.
    cleaned_df = cleaned_df.reset_index(drop=True)
    stats["final_rows"] = len(cleaned_df)

    return cleaned_df, stats
