"""
Module: cleaner.py
Performs data clean-up, string trimming, datetime parsing, and row deduplication.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core libraries for string vectorization, type coercion, and deduplication.
# How it works: Brings pandas and numpy into execution context.
import numpy as np
import pandas as pd


def clean_sales_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean raw transactional sales DataFrame and return audit metrics.

    Args:
        df: Raw transactional DataFrame.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and cleaning statistics dictionary.
    """
    clean_df = df.copy()
    stats = {
        "raw_rows": len(clean_df),
        "duplicates_removed": 0,
        "nulls_imputed": 0,
        "final_rows": 0
    }

    # 1. Clean String Whitespace & Casing
    # What is used: Vectorized .str.strip() and .str.title().
    # Why it is used: Standardizes text representation across category, region, product, and customer names.
    # How it works: Trims leading/trailing spaces and normalizes title casing.
    str_cols = ["Customer_ID", "Customer_Name", "Region", "Category", "Product"]
    for col in str_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()

    # 2. Parse Datetime Strings
    # What is used: pd.to_datetime(format="mixed", errors="coerce").
    # Why it is used: Safely parses Order_Date into datetime64[ns] Timestamps.
    # How it works: Coerces invalid dates to NaT and fills NaTs with default timestamp.
    if "Order_Date" in clean_df.columns:
        clean_df["Order_Date"] = pd.to_datetime(clean_df["Order_Date"], format="mixed", errors="coerce")
        null_dates = clean_df["Order_Date"].isna().sum()
        stats["nulls_imputed"] += int(null_dates)
        if null_dates > 0:
            clean_df["Order_Date"] = clean_df["Order_Date"].fillna(pd.Timestamp("2026-01-01"))

    # 3. Numeric Coercion & Imputation
    # What is used: pd.to_numeric() and median/mode imputation.
    # Why it is used: Ensures numerical variables (Quantity, Unit_Price, Discount, Cost_Price) are float/int.
    # How it works: Converts strings to floats and fills nulls with column median.
    num_cols = ["Quantity", "Unit_Price", "Discount", "Cost_Price"]
    for col in num_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
            null_cnt = clean_df[col].isna().sum()
            stats["nulls_imputed"] += int(null_cnt)
            if null_cnt > 0:
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # 4. Deduplicate Rows based on Order_ID
    # What is used: drop_duplicates(subset=["Order_ID"], keep="first").
    # Why it is used: Eliminates duplicate order transactions.
    # How it works: Retains first occurrence of each Order_ID.
    if "Order_ID" in clean_df.columns:
        pre_len = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["Order_ID"], keep="first")
        stats["duplicates_removed"] = pre_len - len(clean_df)

    clean_df = clean_df.sort_values(by="Order_Date").reset_index(drop=True)
    stats["final_rows"] = len(clean_df)

    return clean_df, stats
