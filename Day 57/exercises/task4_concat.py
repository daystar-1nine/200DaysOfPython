"""
Day 57 - Practical Task 4: DataFrame Concatenation
Demonstrates combining monthly sales DataFrames vertically using pd.concat().
"""

# What is used: Import pandas library.
# Why it is used: Core package for pd.concat operations.
# How it works: Imports pandas namespace.
import pandas as pd


def combine_monthly_sales(jan_df: pd.DataFrame, feb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate January and February sales DataFrames and reset row index.

    Args:
        jan_df: January sales DataFrame.
        feb_df: February sales DataFrame.

    Returns:
        pd.DataFrame: Combined sales DataFrame.
    """
    # What is used: pd.concat([jan_df, feb_df], axis=0, ignore_index=True).
    # Why it is used: Stacks rows vertically and resets index sequence continuously.
    # How it works: Aligns columns and appends rows from feb_df below jan_df.
    combined = pd.concat([jan_df, feb_df], axis=0, ignore_index=True)
    return combined


if __name__ == "__main__":
    # What is used: Sample dictionaries for January and February sales.
    # Why it is used: Simulates separate monthly reporting files.
    # How it works: Maps monthly transaction fields.
    jan_data = {
        "Order_ID": [1001, 1002],
        "Month": ["January", "January"],
        "Sales": [800, 500]
    }
    feb_data = {
        "Order_ID": [1003, 1004],
        "Month": ["February", "February"],
        "Sales": [600, 900]
    }

    # What is used: pd.DataFrame constructors.
    # Why it is used: Creates two distinct monthly DataFrames.
    # How it works: Initializes tabular DataFrames.
    jan_df = pd.DataFrame(jan_data)
    feb_df = pd.DataFrame(feb_data)

    # What is used: Calling combine_monthly_sales.
    # Why it is used: Concatenates DataFrames into a single combined view.
    # How it works: Displays concatenated DataFrame.
    total_sales = combine_monthly_sales(jan_df, feb_df)
    print("--- Combined Monthly Sales Data ---")
    print(total_sales)
