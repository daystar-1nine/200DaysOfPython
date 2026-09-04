"""
Day 58 - Coding Challenge 5: Reusable Data-Cleaning Pipeline Function
Demonstrates clean_dataset(df) function combining all data cleaning steps into a single reusable pipeline.
"""

# What is used: Import sys, pandas, and numpy libraries.
# Why it is used: Core libraries for system encoding configuration, full dataset cleaning, and preprocessing.
# How it works: Brings sys, pandas, and numpy namespaces into scope.
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def clean_dataset(
    df: pd.DataFrame,
    id_col: str = "customer_id",
    numeric_cols: list[str] = None,
    date_cols: list[str] = None,
    text_cols: list[str] = None
) -> pd.DataFrame:
    """
    Comprehensive, reusable data-cleaning pipeline that standardizes column headers,
    trims string whitespace, parses monetary/numeric fields, converts datetime columns,
    normalizes categories, removes duplicate records, validates domain ranges, and imputes missing values.

    Args:
        df: Input raw messy DataFrame.
        id_col: Primary key ID column header (default 'customer_id').
        numeric_cols: List of numeric column names.
        date_cols: List of datetime column names.
        text_cols: List of text column names.

    Returns:
        pd.DataFrame: Cleaned, analysis-ready DataFrame.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures raw input DataFrame is preserved immutably.
    # How it works: Duplicates DataFrame memory buffer.
    clean_df = df.copy()

    # 1. Clean Column Headers (lowercase, stripped, underscores)
    # What is used: String manipulations on df.columns.
    # Why it is used: Normalizes column names into clean Python identifiers.
    # How it works: Converts header string list to lower case and replaces spaces with underscores.
    clean_df.columns = (
        clean_df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Resolve column names after standardization
    numeric_cols = numeric_cols or ["age", "salary"]
    date_cols = date_cols or ["join_date"]
    text_cols = text_cols or ["name", "city", "department"]

    # 2. Clean String Columns (trim whitespace, title case)
    # What is used: Series.astype(str).str.strip().str.title().
    # Why it is used: Normalizes text casing and removes leading/trailing spaces.
    # How it works: Applies string operations across specified text columns.
    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = (
                clean_df[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace("Nan", np.nan)
                .replace("None", np.nan)
            )

    # 3. Standardize Categorical Gender Column if present
    # What is used: Dictionary mapping and string lower-casing.
    # Why it is used: Standardizes gender representations.
    # How it works: Maps 'm', 'male' -> 'Male', 'f', 'female' -> 'Female'.
    if "gender" in clean_df.columns:
        g_clean = clean_df["gender"].astype(str).str.strip().str.lower()
        g_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
        clean_df["gender"] = g_clean.map(g_map).fillna("Unknown")

    # 4. Clean & Convert Numeric Columns
    # What is used: Currency symbol replacement and pd.to_numeric(errors="coerce").
    # Why it is used: Converts monetary and numeric string columns to float numbers safely.
    # How it works: Replaces currency symbols and converts unparseable strings to NaNs.
    for col in numeric_cols:
        if col in clean_df.columns:
            cleaned_str = (
                clean_df[col]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            clean_df[col] = pd.to_numeric(cleaned_str, errors="coerce")

    # 5. Clean & Convert Date Columns
    # What is used: pd.to_datetime(errors="coerce").
    # Why it is used: Parses date strings into Timestamps, coercing invalid entries to NaT.
    # How it works: Converts date formats to datetime64[ns].
    for col in date_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")

    # 6. Validate Domain Ranges (Age: 0-120, Salary >= 0)
    # What is used: Conditional assignment via df.loc.
    # Why it is used: Sets out-of-bounds invalid values to NaN.
    # How it works: Checks age and salary boundary conditions.
    if "age" in clean_df.columns:
        clean_df.loc[~clean_df["age"].between(0, 120), "age"] = np.nan
    if "salary" in clean_df.columns:
        clean_df.loc[clean_df["salary"] < 0, "salary"] = np.nan

    # 7. Remove Duplicate Rows based on Primary ID Key
    # What is used: df.drop_duplicates(subset=[id_col], keep="first").
    # Why it is used: Eliminates duplicate customer records.
    # How it works: Retains first occurrence of primary ID.
    if id_col in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=[id_col], keep="first")
    else:
        clean_df = clean_df.drop_duplicates(keep="first")

    # 8. Impute Missing Values (Median for numeric, Mode/Unknown for text)
    # What is used: Series.fillna() with column median and mode.
    # Why it is used: Fills remaining missing data safely based on statistical properties.
    # How it works: Calculates median/mode ignoring NaNs and fills missing slots.
    for col in numeric_cols:
        if col in clean_df.columns:
            med = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(round(float(med), 1) if not pd.isna(med) else 0.0)

    for col in text_cols:
        if col in clean_df.columns:
            mode_vals = clean_df[col].mode()
            fill_val = str(mode_vals[0]) if len(mode_vals) > 0 else "Unknown"
            clean_df[col] = clean_df[col].fillna(fill_val)

    # 9. Reset Index
    # What is used: reset_index(drop=True).
    # Why it is used: Re-indexes cleaned DataFrame continuously.
    # How it works: Generates fresh 0..N-1 index.
    clean_df = clean_df.reset_index(drop=True)
    return clean_df


if __name__ == "__main__":
    # What is used: Raw messy DataFrame with all types of data defects.
    # Why it is used: Input dataset to demonstrate clean_dataset reusable pipeline.
    # How it works: Holds missing entries, extra spaces, currency symbols, invalid age, and duplicate IDs.
    messy_df = pd.DataFrame({
        " Customer_ID ": ["C01", "C02", "C01", "C03", "C04"],
        " Name ": [" rahul sawant ", "PRIYA PATEL", " rahul sawant ", "aman verma", "sneha kulkarni"],
        " Age ": ["20", " -5 ", "20", "150", " 25 "],
        " Gender ": [" M ", "female", " M ", "MALE", "F"],
        " Salary ": ["₹50,000", "₹60,000", "₹50,000", "unknown", "-10000"],
        " City ": [" mumbai ", "PUNE", " mumbai ", "DELHI", "mumbai"],
        " Join_Date ": ["01-01-2026", "2026/02/10", "01-01-2026", "March 5, 2026", "invalid"]
    })

    print("--- Raw Messy Input DataFrame ---")
    print(messy_df)

    # What is used: Calling clean_dataset.
    # Why it is used: Runs end-to-end reusable data cleaning pipeline.
    # How it works: Returns clean, analysis-ready DataFrame.
    cleaned_df = clean_dataset(messy_df, id_col="customer_id")

    print("\n--- Cleaned & Processed DataFrame ---")
    print(cleaned_df)
    print("\nData Types:")
    print(cleaned_df.dtypes)
