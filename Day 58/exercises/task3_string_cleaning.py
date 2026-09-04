"""
Day 58 - Practical Task 3: String Normalization & Whitespace Trimming
Demonstrates stripping whitespace and title-casing messy city names (mUMbAi, DELHI, pune -> Mumbai, Delhi, Pune).
"""

# What is used: Import pandas library.
# Why it is used: Core package for vectorized string operations.
# How it works: Brings pandas namespace into execution context.
import pandas as pd


def clean_city_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean messy City string column by stripping whitespace and applying Title Case.

    Args:
        df: Input DataFrame containing City column.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures original input DataFrame is preserved.
    # How it works: Duplicates DataFrame memory buffer.
    res_df = df.copy()

    # What is used: Series.astype(str).str.strip().str.title().
    # Why it is used: Strips accidental leading/trailing whitespace and normalizes text casing.
    # How it works: Converts values to strings, removes whitespace, and capitalizes first letter of each word.
    res_df["City"] = res_df["City"].astype(str).str.strip().str.title()
    return res_df


if __name__ == "__main__":
    # What is used: Dictionary with messy city string variations.
    # Why it is used: Input dataset for string normalization test.
    # How it works: Contains whitespace and irregular capitalization.
    messy_data = {
        "Name": ["Rahul", "Priya", "Aman"],
        "City": ["  mUMbAi ", "DELHI", " pune "]
    }
    df = pd.DataFrame(messy_data)

    # What is used: Calling clean_city_strings.
    # Why it is used: Normalizes City strings.
    # How it works: Prints cleaned DataFrame containing standardized city names.
    clean_df = clean_city_strings(df)
    print("--- Normalized City Strings ---")
    print(clean_df)
