"""
Day 56 - Coding Challenge 3: Grade Column Assignment
Assign letter grades (A+, A, B, C, F) to students based on average marks using pd.cut().
"""

# What is used: Import pandas library.
# Why it is used: Essential for pd.cut binning and DataFrame operations.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def assign_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average mark and assign letter grades to each student.

    Bins: [0, 50, 70, 85, 90, 100]
    Labels: ['F', 'C', 'B', 'A', 'A+']

    Args:
        df: Input DataFrame with Math, Physics, and Chemistry scores.

    Returns:
        pd.DataFrame: DataFrame augmented with Average and Grade columns.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Prevents mutating original input DataFrame parameter.
    # How it works: Allocates new memory block holding copied data.
    result_df = df.copy()

    # What is used: Vectorized arithmetic division on row sum.
    # Why it is used: Computes average mark across 3 subjects for every row.
    # How it works: Sums subject Series elementwise and divides by scalar 3.0.
    result_df["Average"] = (result_df["Math"] + result_df["Physics"] + result_df["Chemistry"]) / 3.0

    # What is used: pd.cut() binning function.
    # Why it is used: Categorizes continuous numeric values into discrete letter grade bins.
    # How it works: Evaluates each average score against bin intervals and assigns matching label.
    bins = [0, 50, 70, 85, 90, 100]
    labels = ["F", "C", "B", "A", "A+"]
    result_df["Grade"] = pd.cut(result_df["Average"], bins=bins, labels=labels, include_lowest=True)

    return result_df


if __name__ == "__main__":
    # What is used: Mock dictionary of student marks.
    # Why it is used: Standardized sample dataset for grading logic test.
    # How it works: Maps student records into dictionary format.
    sample = {
        "Name": ["Aarav", "Kabir", "Rohan", "Tanvi"],
        "Math": [95, 45, 78, 88],
        "Physics": [96, 52, 82, 85],
        "Chemistry": [94, 48, 80, 87]
    }

    # What is used: pd.DataFrame creation.
    # Why it is used: Constructs DataFrame from dictionary.
    # How it works: Initializes tabular structure.
    df = pd.DataFrame(sample)

    # What is used: assign_grades function call.
    # Why it is used: Appends Average and Grade columns.
    # How it works: Computes scores and prints result table.
    graded_df = assign_grades(df)
    print("--- Student Grade Assignments ---")
    print(graded_df[["Name", "Average", "Grade"]])
