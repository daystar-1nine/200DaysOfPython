"""
Day 56 - Coding Challenge 4: Subject Average Marks Calculator
Calculate average marks for each subject handling missing NaN values safely.
"""

# What is used: Import pandas and numpy libraries.
# Why it is used: Pandas for Series stats, NumPy for np.nan placeholder representation.
# How it works: Loads pandas and numpy modules.
import numpy as np
import pandas as pd


def compute_subject_averages(df: pd.DataFrame) -> pd.Series:
    """
    Compute mean mark per subject, ignoring NaN values.

    Args:
        df: Input DataFrame containing numeric subject columns.

    Returns:
        pd.Series: Series containing mean marks keyed by subject name.
    """
    # What is used: df.mean(numeric_only=True) aggregation method.
    # Why it is used: Automatically ignores NaN values (skipna=True default) and calculates column means.
    # How it works: Sums valid numerical elements per column and divides by non-null row count.
    subject_cols = ["Math", "Physics", "Chemistry"]
    return df[subject_cols].mean()


if __name__ == "__main__":
    # What is used: Dictionary containing missing values represented by np.nan.
    # Why it is used: Simulates real-world incomplete student datasets.
    # How it works: Holds nan float values in Math and Physics columns.
    raw_data = {
        "Name": ["Aarav", "Yash", "Riya", "Dev"],
        "Math": [85.0, np.nan, 79.0, 88.0],
        "Physics": [90.0, 80.0, np.nan, 85.0],
        "Chemistry": [92.0, 82.0, 85.0, np.nan]
    }

    # What is used: pd.DataFrame creation.
    # Why it is used: Instantiates DataFrame containing missing entries.
    # How it works: Converts dictionary into 2D table.
    df = pd.DataFrame(raw_data)

    # What is used: compute_subject_averages invocation.
    # Why it is used: Computes robust subject-wise averages ignoring NaNs.
    # How it works: Returns pandas Series with average score per subject.
    averages = compute_subject_averages(df)
    print("--- Subject Averages (Ignoring NaNs) ---")
    print(averages.round(2))
