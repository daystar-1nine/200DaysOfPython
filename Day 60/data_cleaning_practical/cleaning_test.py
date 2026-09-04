"""
Day 60 - Data Cleaning Practical Test
Cleans deliberately messy DataFrame: whitespace stripping, title casing, numeric age coercion, out-of-bounds invalidation, and deduplication.
"""

# What is used: Import sys, numpy, and pandas modules.
# Why it is used: Cross-platform output encoding and tabular data cleaning.
# How it works: Brings sys, numpy, and pandas namespaces into scope.
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def clean_messy_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Execute full data cleaning operations on messy test dataset.

    Args:
        df: Deliberately messy input DataFrame.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and audit report.
    """
    clean_df = df.copy()

    # 1. Clean String Columns (Name, City)
    clean_df["Name"] = clean_df["Name"].astype(str).str.strip().str.title()
    clean_df["City"] = clean_df["City"].astype(str).str.strip().str.title()

    # 2. Convert Age to numeric & Invalidate Out-of-Bounds
    clean_df["Age"] = pd.to_numeric(clean_df["Age"], errors="coerce")
    invalid_mask = (~clean_df["Age"].between(0, 120)) & clean_df["Age"].notna()
    clean_df.loc[invalid_mask, "Age"] = np.nan

    # 3. Detect and Remove Duplicates
    dup_count = int(clean_df.duplicated().sum())
    clean_df = clean_df.drop_duplicates(keep="first").reset_index(drop=True)

    # 4. Impute missing age with median
    med_age = clean_df["Age"].median()
    clean_df["Age"] = clean_df["Age"].fillna(med_age)

    # 5. Validation Check
    is_valid = (
        clean_df["Age"].between(0, 120).all()
        and not clean_df.duplicated().any()
        and clean_df.isna().sum().sum() == 0
    )

    audit = {
        "initial_rows": len(df),
        "duplicates_removed": dup_count,
        "final_rows": len(clean_df),
        "is_valid": is_valid
    }

    return clean_df, audit


def main() -> None:
    raw_df = pd.DataFrame({
        "Name": [
            " Rahul ",
            "PRIYA",
            " aman ",
            "Sneha",
            "Sneha"
        ],
        "Age": [
            "20",
            "21",
            "-5",
            "twenty",
            "22"
        ],
        "City": [
            "Mumbai",
            "mumbai ",
            "PUNE",
            " Delhi ",
            " Delhi "
        ]
    })

    print("--- Raw Messy DataFrame ---")
    print(raw_df)

    clean_df, audit = clean_messy_dataframe(raw_df)

    print("\n--- Cleaned & Validated DataFrame ---")
    print(clean_df)
    print("\n--- Audit Summary ---")
    print(audit)


if __name__ == "__main__":
    main()
