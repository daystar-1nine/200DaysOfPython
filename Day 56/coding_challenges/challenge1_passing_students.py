"""
Day 56 - Coding Challenge 1: Passing Students Filter
Filter students who have passed all subjects (marks >= 50 in Math, Physics, and Chemistry).
"""

# What is used: Import pandas library for DataFrame operations.
# Why it is used: Essential for tabular data manipulation and boolean filtering.
# How it works: Loads the pandas package into the Python workspace.
import pandas as pd


def filter_passing_students(df: pd.DataFrame, pass_mark: float = 50.0) -> pd.DataFrame:
    """
    Filter students who scored greater than or equal to pass_mark in all subjects.

    Args:
        df: Input DataFrame containing Math, Physics, and Chemistry columns.
        pass_mark: Minimum passing threshold (default 50.0).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only passing students.
    """
    # What is used: Boolean mask using bitwise AND (&) operator across multiple conditions.
    # Why it is used: Vectorized element-wise filtering in Pandas requires bitwise operators.
    # How it works: Evaluates three boolean Series and returns True only when all three are True.
    mask = (df["Math"] >= pass_mark) & (df["Physics"] >= pass_mark) & (df["Chemistry"] >= pass_mark)
    
    # What is used: Label-based bracket indexing df[mask].
    # Why it is used: Filters rows based on the boolean condition.
    # How it works: Retains only rows where mask values are True.
    return df[mask].copy()


if __name__ == "__main__":
    # What is used: Dictionary defining sample student dataset.
    # Why it is used: Provides raw tabular data to construct a test DataFrame.
    # How it works: Maps column header strings to lists of values.
    sample_data = {
        "Student_ID": ["S01", "S02", "S03", "S04"],
        "Name": ["Aarav", "Ananya", "Rohan", "Priya"],
        "Math": [85, 45, 78, 95],
        "Physics": [90, 52, 42, 96],
        "Chemistry": [92, 48, 80, 94]
    }

    # What is used: pd.DataFrame constructor.
    # Why it is used: Converts dictionary into a 2D Pandas DataFrame.
    # How it works: Aligns keys as columns and lists as row entries.
    df_students = pd.DataFrame(sample_data)

    # What is used: Function call filter_passing_students.
    # Why it is used: Extracts students meeting the pass criteria.
    # How it works: Returns filtered DataFrame of passing students.
    passing_df = filter_passing_students(df_students, pass_mark=50.0)
    print("--- Passing Students ---")
    print(passing_df)
