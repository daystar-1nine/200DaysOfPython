"""
Day 58 - Practical Task 2: Duplicate Detection & Removal
Demonstrates detecting duplicate rows using df.duplicated() and removing them with df.drop_duplicates().
"""

# What is used: Import pandas library.
# Why it is used: Core package for duplicate detection and deduplication.
# How it works: Brings pandas namespace into module scope.
import pandas as pd


def process_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    """
    Detect duplicate rows, count repetitions, and return clean deduplicated DataFrame.

    Args:
        df: Input DataFrame containing potential duplicate records.

    Returns:
        tuple[pd.DataFrame, pd.Series, int]: Deduplicated DataFrame, boolean duplicate Series, duplicate count.
    """
    # What is used: df.duplicated().
    # Why it is used: Identifies repeated rows in DataFrame.
    # How it works: Returns boolean Series marking True for duplicate occurrences.
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())

    # What is used: df.drop_duplicates(keep="first").
    # Why it is used: Removes duplicate rows, keeping the first occurrence.
    # How it works: Filters out repeated rows.
    clean_df = df.drop_duplicates(keep="first").reset_index(drop=True)

    return clean_df, dup_mask, dup_count


if __name__ == "__main__":
    # What is used: Dictionary containing intentional duplicate rows.
    # Why it is used: Serves as test data for duplicate detection.
    # How it works: Contains repeated entries for Rahul and Priya.
    data = {
        "Name": ["Rahul", "Priya", "Rahul", "Aman", "Priya"],
        "Age": [20, 21, 20, 22, 21],
        "City": ["Mumbai", "Pune", "Mumbai", "Delhi", "Pune"]
    }
    df = pd.DataFrame(data)

    # What is used: Calling process_duplicates.
    # Why it is used: Detects and drops duplicate rows.
    # How it works: Prints total duplicate count and deduplicated table.
    clean_df, dup_mask, count = process_duplicates(df)

    print(f"Total Duplicate Rows Found: {count}")
    print("\n--- Deduplicated DataFrame ---")
    print(clean_df)
