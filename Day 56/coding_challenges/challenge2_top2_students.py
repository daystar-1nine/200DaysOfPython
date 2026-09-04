"""
Day 56 - Coding Challenge 2: Top 2 Scoring Students
Identify the top 2 overall students based on total marks.
"""

# What is used: Import pandas library.
# Why it is used: Required for DataFrame ranking and nlargest calculations.
# How it works: Imports pandas module into namespace.
import pandas as pd


def get_top_n_students(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Calculate total marks and return top n students based on Total score.

    Args:
        df: DataFrame containing student marks.
        n: Number of top students to return.

    Returns:
        pd.DataFrame: Top n performing students sorted by Total descending.
    """
    # What is used: Vectorized column addition df["Math"] + df["Physics"] + df["Chemistry"].
    # Why it is used: Efficiently computes row-wise sum across subject columns.
    # How it works: Adds elements positionally across Series to create a new "Total" column.
    df_copy = df.copy()
    df_copy["Total"] = df_copy["Math"] + df_copy["Physics"] + df_copy["Chemistry"]

    # What is used: df.nlargest() method.
    # Why it is used: Quickly finds the top n rows based on a specific column without full sorting.
    # How it works: Sorts internal binary heap and returns top n rows ordered by "Total".
    return df_copy.nlargest(n, "Total")


if __name__ == "__main__":
    # What is used: Dictionary of test data.
    # Why it is used: Serves as mock input to demonstrate function operation.
    # How it works: Defines student scores for 4 candidates.
    data = {
        "Name": ["Aarav", "Ananya", "Rohan", "Priya"],
        "Math": [85, 92, 78, 95],
        "Physics": [90, 88, 82, 96],
        "Chemistry": [92, 95, 80, 94]
    }

    # What is used: pd.DataFrame creation.
    # Why it is used: Prepares tabular dataset for analysis.
    # How it works: Converts dictionary into structured DataFrame.
    df = pd.DataFrame(data)

    # What is used: Calling get_top_n_students function.
    # Why it is used: Obtains the top 2 scoring students.
    # How it works: Returns DataFrame slice containing top 2 candidates.
    top2 = get_top_n_students(df, n=2)
    print("--- Top 2 Students ---")
    print(top2[["Name", "Total"]])
