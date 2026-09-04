"""
Module: cleaner.py
Performs automated data cleaning, whitespace stripping, date parsing, and deduplication for Day 60 BI Analytics Engine.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core libraries for string vectorization, type coercion, and deduplication.
# How it works: Cleans string columns, coerces dates, casts numbers, and removes duplicate orders.
import numpy as np
import pandas as pd


def clean_sales_records(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Execute full cleaning pipeline on raw transactional sales data.

    Args:
        df: Raw input DataFrame.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and audit report dictionary.
    """
    clean_df = df.copy()
    audit = {
        "raw_rows": len(clean_df),
        "duplicates_removed": 0,
        "nulls_filled": 0,
        "dates_coerced": 0,
        "final_rows": 0
    }

    # 1. String Columns Trimming & Standardizing
    str_cols = ["Order_ID", "Customer_ID", "Customer_Name", "Region", "Category", "Product"]
    for col in str_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()

    # 2. Parse Order_Date to Datetime
    if "Order_Date" in clean_df.columns:
        clean_df["Order_Date"] = pd.to_datetime(clean_df["Order_Date"], format="mixed", errors="coerce")
        nat_count = int(clean_df["Order_Date"].isna().sum())
        audit["dates_coerced"] = nat_count
        if nat_count > 0:
            clean_df["Order_Date"] = clean_df["Order_Date"].fillna(pd.Timestamp("2026-01-01"))

    # 3. Numeric Coercion & Imputation
    num_cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount"]
    for col in num_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
            null_cnt = int(clean_df[col].isna().sum())
            audit["nulls_filled"] += null_cnt
            if null_cnt > 0:
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # 4. Deduplicate on Order_ID
    if "Order_ID" in clean_df.columns:
        pre_len = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["Order_ID"], keep="first")
        audit["duplicates_removed"] = pre_len - len(clean_df)

    clean_df = clean_df.sort_values(by="Order_Date").reset_index(drop=True)
    audit["final_rows"] = len(clean_df)

    return clean_df, audit
