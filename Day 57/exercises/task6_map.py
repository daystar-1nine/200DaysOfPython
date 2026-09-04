"""
Day 57 - Practical Task 6: Value Mapping with map()
Demonstrates mapping department acronym codes to full department names using map().
"""

# What is used: Import pandas library.
# Why it is used: Core package for Series.map operations.
# How it works: Imports pandas namespace into execution context.
import pandas as pd


def map_department_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Department codes (CSE, DS, ECE) to full department names.

    Args:
        df: Input DataFrame with Department column.

    Returns:
        pd.DataFrame: Augmented DataFrame with Department_Name column.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Prevents side-effects on original input DataFrame.
    # How it works: Duplicates DataFrame memory buffer.
    res_df = df.copy()

    # What is used: Dictionary mapping acronyms to full strings.
    # Why it is used: Lookup dictionary for value transformation.
    # How it works: Maps 'CSE' -> 'Computer Science', etc.
    mapping = {
        "CSE": "Computer Science",
        "DS": "Data Science",
        "ECE": "Electronics"
    }

    # What is used: res_df["Department"].map(mapping).
    # Why it is used: Transforms Series elements using dictionary lookup table.
    # How it works: Replaces key occurrences with corresponding dictionary values.
    res_df["Department_Name"] = res_df["Department"].map(mapping)
    return res_df


if __name__ == "__main__":
    # What is used: Mock dictionary of student department codes.
    # Why it is used: Input dataset for testing Series.map().
    # How it works: Maps student names to department acronyms.
    sample = {
        "Name": ["Aarav", "Ananya", "Rohan", "Priya"],
        "Department": ["CSE", "DS", "ECE", "CSE"]
    }
    df = pd.DataFrame(sample)

    # What is used: Calling map_department_names.
    # Why it is used: Performs dictionary mapping on Department column.
    # How it works: Prints augmented DataFrame with full department names.
    mapped_df = map_department_names(df)
    print("--- Department Name Mapping ---")
    print(mapped_df)
