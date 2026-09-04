"""
Day 58 - Practical Task 4: Monetary String Cleaning & Numeric Conversion
Demonstrates parsing currency strings (₹50,000, 60000, ₹75,500, unknown) into float numbers.
"""

# What is used: Import pandas library.
# Why it is used: Core package for string replace and pd.to_numeric conversion.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def clean_monetary_column(df: pd.DataFrame, col_name: str = "Price") -> pd.DataFrame:
    """
    Clean monetary string column by stripping currency symbols (₹, $) and commas,
    then coercing values to numeric float.

    Args:
        df: Input DataFrame containing monetary string column.
        col_name: Column header to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame with numeric float column.
    """
    # What is used: df.copy() method.
    # Why it is used: Preserves input DataFrame without mutating original buffer.
    # How it works: Duplicates DataFrame.
    res_df = df.copy()

    # What is used: String replacements .str.replace("₹", "").str.replace(",", "").str.strip().
    # Why it is used: Removes currency symbols and comma separators.
    # How it works: Replaces unwanted character tokens with empty strings.
    clean_str = (
        res_df[col_name]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # What is used: pd.to_numeric(clean_str, errors="coerce").
    # Why it is used: Safely parses valid numeric strings to float, mapping unparseable values (e.g. 'unknown') to NaN.
    # How it works: Returns float Series with NaNs for non-numeric tokens.
    res_df[col_name] = pd.to_numeric(clean_str, errors="coerce")

    return res_df


if __name__ == "__main__":
    # What is used: Dictionary containing currency string formatting.
    # Why it is used: Test data for monetary string parsing.
    # How it works: Includes ₹ symbols, commas, and unparseable 'unknown' string.
    sample = {
        "Item": ["A", "B", "C", "D"],
        "Price": ["₹50,000", "60000", "₹75,500", "unknown"]
    }
    df = pd.DataFrame(sample)

    # What is used: Calling clean_monetary_column.
    # Why it is used: Converts Price column to numeric floats.
    # How it works: Displays cleaned DataFrame with float Price column and NaN for 'unknown'.
    cleaned_df = clean_monetary_column(df, "Price")
    print("--- Cleaned Monetary Price Column ---")
    print(cleaned_df)
    print("\nData Types:")
    print(cleaned_df.dtypes)
