"""
Day 56 - Practical Task 2: DataFrame Analytics & Inspection
Demonstrates DataFrame creation, df.info(), df.describe(), loc/iloc selection, and shape analysis.
"""

# What is used: Import pandas library.
# Why it is used: Core package for DataFrame tabular analytics.
# How it works: Loads pandas module.
import pandas as pd


def analyze_student_dataframe() -> pd.DataFrame:
    """
    Create a student DataFrame and perform inspection and structural analysis.

    Returns:
        pd.DataFrame: Created student DataFrame.
    """
    # What is used: Dictionary containing multi-column student dataset.
    # Why it is used: Provides raw structured input data.
    # How it works: Maps column strings to lists of values.
    data = {
        "Student_ID": ["S101", "S102", "S103", "S104", "S105"],
        "Name": ["Aarav", "Ananya", "Rohan", "Priya", "Vikram"],
        "Department": ["CSE", "DS", "ECE", "CSE", "DS"],
        "Math": [85, 92, 78, 95, 60],
        "Physics": [90, 88, 82, 96, 65],
        "Chemistry": [92, 95, 80, 94, 58]
    }

    # What is used: pd.DataFrame constructor.
    # Why it is used: Instantiates 2D heterogeneous tabular object.
    # How it works: Sets up column dtypes and index structures.
    df = pd.DataFrame(data)

    print("--- DataFrame Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- DataFrame Column Types & Non-Null Counts ---")
    print(df.dtypes)

    print("\n--- Summary Statistics (Numeric Columns) ---")
    print(df.describe())

    print("\n--- Value Counts by Department ---")
    print(df["Department"].value_counts())

    # What is used: Selection with .loc vs .iloc.
    # Why it is used: Demonstrates label vs positional indexing.
    # How it works: loc accesses by index label & column name; iloc accesses by integer index position.
    first_student_name = df.loc[0, "Name"]
    second_student_math = df.iloc[1, 3]

    print(f"\nFirst Student (loc[0, 'Name']): {first_student_name}")
    print(f"Second Student Math (iloc[1, 3]): {second_student_math}")

    return df


if __name__ == "__main__":
    analyze_student_dataframe()
