"""
Module: cleaner.py
Performs data cleaning, deduplication, missing value imputation, and boundary validation pipelines.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core libraries for DataFrame cleaning and numerical type coercion.
# How it works: Brings pandas and numpy into execution context.
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean raw sales DataFrame by dropping duplicate Order_IDs, imputing NaNs,
    validating numeric ranges (Quantity > 0, Price > 0, Discount between 0 and 1),
    and stripping string whitespace.

    Args:
        df: Raw sales DataFrame.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and audit statistics dictionary.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures raw input DataFrame is preserved without side-effects.
    # How it works: Allocates new memory block holding copied data.
    cleaned_df = df.copy()
    stats = {
        "initial_rows": len(cleaned_df),
        "duplicates_removed": 0,
        "nulls_filled": 0,
        "invalid_rows_dropped": 0,
        "final_rows": 0
    }

    # What is used: pd.api.types.is_string_dtype and .str.strip().
    # Why it is used: Strips accidental whitespace from text columns.
    # How it works: Applies str.strip() across string columns.
    str_cols = ["Order_ID", "Customer_ID", "Customer_Name", "Region", "Category", "Product"]
    for col in str_cols:
        if col in cleaned_df.columns and pd.api.types.is_string_dtype(cleaned_df[col]):
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # What is used: df.drop_duplicates(subset=["Order_ID"], keep="first").
    # Why it is used: Removes duplicate transactions based on primary key Order_ID.
    # How it works: Keeps first occurrence of each Order_ID and drops subsequent duplicates.
    pre_dedupe = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["Order_ID"], keep="first")
    stats["duplicates_removed"] = pre_dedupe - len(cleaned_df)

    # What is used: pd.to_numeric() with errors='coerce'.
    # Why it is used: Ensures Quantity, Unit_Price, and Discount are float/int types.
    # How it works: Converts valid numeric strings to numbers and unparseable values to NaN.
    num_cols = ["Quantity", "Unit_Price", "Discount"]
    for col in num_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    # What is used: Imputation with median value.
    # Why it is used: Safely fills missing numeric values using column median.
    # How it works: Calculates column median ignoring NaNs and fills NaNs.
    null_count = int(cleaned_df[num_cols].isnull().sum().sum())
    stats["nulls_filled"] = null_count

    for col in num_cols:
        med = cleaned_df[col].median()
        if pd.isna(med):
            med = 0.0 if col == "Discount" else 1.0
        cleaned_df[col] = cleaned_df[col].fillna(med)

    # What is used: Vectorized boolean masking for valid numerical ranges.
    # Why it is used: Filters out corrupted or invalid records (Quantity > 0, Price > 0, 0 <= Discount <= 1).
    # How it works: Combines boolean conditions across numeric Series.
    valid_mask = (
        (cleaned_df["Quantity"] > 0)
        & (cleaned_df["Unit_Price"] > 0)
        & (cleaned_df["Discount"] >= 0.0)
        & (cleaned_df["Discount"] <= 1.0)
    )

    pre_valid = len(cleaned_df)
    cleaned_df = cleaned_df[valid_mask].copy()
    stats["invalid_rows_dropped"] = pre_valid - len(cleaned_df)

    # What is used: reset_index(drop=True).
    # Why it is used: Generates continuous 0 to N-1 integer index after row drops.
    # How it works: Discards old sparse index labels.
    cleaned_df = cleaned_df.reset_index(drop=True)
    stats["final_rows"] = len(cleaned_df)

    return cleaned_df, stats
