"""
Day 58 - Practical Task 5: Messy Date Parsing & NaT Handling
Demonstrates parsing messy date strings (01-01-2026, 2026/02/10, March 5, 2026, invalid) into datetime objects.
"""

# What is used: Import pandas library.
# Why it is used: Core package for pd.to_datetime parsing and NaT handling.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def clean_date_strings(df: pd.DataFrame, date_col: str = "Order_Date") -> tuple[pd.DataFrame, int]:
    """
    Parse date string column into datetime objects, coercing unparseable dates to NaT.

    Args:
        df: Input DataFrame containing messy date strings.
        date_col: Date column header.

    Returns:
        tuple[pd.DataFrame, int]: Transformed DataFrame and count of invalid NaT dates.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures input DataFrame is not mutated.
    # How it works: Duplicates DataFrame.
    res_df = df.copy()

    # What is used: pd.to_datetime(res_df[date_col], errors="coerce").
    # Why it is used: Converts messy date formats to Timestamp objects, coercing invalid entries to NaT.
    # How it works: Parses various string date patterns into datetime64[ns].
    res_df[date_col] = pd.to_datetime(res_df[date_col], errors="coerce")

    # What is used: res_df[date_col].isna().sum().
    # Why it is used: Counts number of unparseable/invalid dates.
    # How it works: Sums NaT boolean occurrences.
    invalid_count = int(res_df[date_col].isna().sum())

    return res_df, invalid_count


if __name__ == "__main__":
    # What is used: Dictionary with multiple messy date format strings.
    # Why it is used: Test dataset for pd.to_datetime parsing.
    # How it works: Holds hyphenated, slashed, month-named, and invalid date strings.
    sample = {
        "ID": [1, 2, 3, 4],
        "Order_Date": ["01-01-2026", "2026/02/10", "March 5, 2026", "invalid"]
    }
    df = pd.DataFrame(sample)

    # What is used: Calling clean_date_strings.
    # Why it is used: Parses dates and flags invalid entries.
    # How it works: Prints parsed DataFrame and invalid date count.
    cleaned_df, invalid_cnt = clean_date_strings(df)
    print(f"Invalid Dates Coerced to NaT: {invalid_cnt}")
    print("\n--- Parsed Datetime DataFrame ---")
    print(cleaned_df)
    print("\nData Types:")
    print(cleaned_df.dtypes)
