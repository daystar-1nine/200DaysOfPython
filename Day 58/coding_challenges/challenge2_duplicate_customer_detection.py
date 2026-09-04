"""
Day 58 - Coding Challenge 2: Duplicate Customer Detection & Retention Strategies
Detect duplicate customer ID records and demonstrate keep='first' vs keep='last' retention options.
"""

# What is used: Import pandas library.
# Why it is used: Core package for subset duplicate detection and deduplication.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def resolve_duplicate_customers(df: pd.DataFrame, key_col: str = "Customer_ID", strategy: str = "first") -> tuple[pd.DataFrame, int]:
    """
    Detect and resolve duplicate records based on primary key column.

    Args:
        df: Input DataFrame containing customer records.
        key_col: Key column to check for duplicates (default 'Customer_ID').
        strategy: Retention strategy ('first' or 'last').

    Returns:
        tuple[pd.DataFrame, int]: Cleaned DataFrame and count of duplicate key records removed.
    """
    # What is used: df.duplicated(subset=[key_col]).sum().
    # Why it is used: Counts number of duplicate key entries.
    # How it works: Checks repeated values in key_col.
    initial_dup_count = int(df.duplicated(subset=[key_col]).sum())

    # What is used: df.drop_duplicates(subset=[key_col], keep=strategy).
    # Why it is used: Retains first or last occurrence of repeated customer IDs.
    # How it works: Filters out repeated rows according to strategy.
    clean_df = df.drop_duplicates(subset=[key_col], keep=strategy).reset_index(drop=True)

    return clean_df, initial_dup_count


if __name__ == "__main__":
    # What is used: Dictionary containing duplicate Customer_IDs with different transaction updates.
    # Why it is used: Test data for duplicate customer ID resolution.
    # How it works: Holds 2 entries for C001 and 2 entries for C002.
    customers = pd.DataFrame({
        "Customer_ID": ["C001", "C002", "C001", "C003", "C002"],
        "Name": ["Rahul S.", "Priya P.", "Rahul Sawant", "Aman V.", "Priya Patel"],
        "City": ["Mumbai", "Pune", "Mumbai", "Delhi", "Pune"]
    })

    # What is used: Calling resolve_duplicate_customers with 'first' and 'last' strategies.
    # Why it is used: Demonstrates retention behavior.
    # How it works: Prints deduplicated DataFrames for both strategies.
    clean_first, count_first = resolve_duplicate_customers(customers, strategy="first")
    clean_last, count_last = resolve_duplicate_customers(customers, strategy="last")

    print(f"Duplicate Customer Records Detected: {count_first}")

    print("\n--- Deduplicated (keep='first') ---")
    print(clean_first)

    print("\n--- Deduplicated (keep='last') ---")
    print(clean_last)
