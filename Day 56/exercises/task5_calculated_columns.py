"""
Day 56 - Practical Task 5: Derived & Calculated Columns
Demonstrates adding Total, Average, Result (Pass/Fail), and Grade columns to a DataFrame.
"""

# What is used: Import pandas library.
# Why it is used: Core package for calculated column creation.
# How it works: Brings pandas namespace into module scope.
import pandas as pd


def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment student DataFrame with Total, Average, Result, and Grade calculated columns.

    Args:
        df: Input DataFrame containing Math, Physics, and Chemistry scores.

    Returns:
        pd.DataFrame: Augmented DataFrame.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Prevents mutating original input DataFrame.
    # How it works: Creates independent DataFrame instance.
    res_df = df.copy()

    # What is used: Vectorized row sum df["Math"] + df["Physics"] + df["Chemistry"].
    # Why it is used: Calculates total score across subjects.
    # How it works: Sums elements across Series.
    res_df["Total"] = res_df["Math"] + res_df["Physics"] + res_df["Chemistry"]

    # What is used: Vectorized division res_df["Total"] / 3.0.
    # Why it is used: Computes average percentage score.
    # How it works: Divides total by float scalar 3.0.
    res_df["Average"] = (res_df["Total"] / 3.0).round(2)

    # What is used: Vectorized boolean condition and pd.Series.map or conditional indexing.
    # Why it is used: Assigns 'Pass' if all subjects >= 50 else 'Fail'.
    # How it works: Evaluates boolean condition vector and sets Pass/Fail strings.
    pass_condition = (res_df["Math"] >= 50) & (res_df["Physics"] >= 50) & (res_df["Chemistry"] >= 50)
    res_df["Result"] = pass_condition.map({True: "Pass", False: "Fail"})

    # What is used: pd.cut() function for letter grade binning.
    # Why it is used: Converts numeric Average scores into discrete letter grades.
    # How it works: Maps Average values to defined bin intervals.
    bins = [0, 50, 70, 85, 90, 100]
    labels = ["F", "C", "B", "A", "A+"]
    res_df["Grade"] = pd.cut(res_df["Average"], bins=bins, labels=labels, include_lowest=True)

    return res_df


if __name__ == "__main__":
    # What is used: Dictionary defining mock student records.
    # Why it is used: Input dataset for testing calculated columns logic.
    # How it works: Maps fields to lists.
    data = {
        "Student_ID": ["S101", "S102", "S103", "S104", "S105"],
        "Name": ["Aarav", "Ananya", "Rohan", "Priya", "Vikram"],
        "Department": ["CSE", "DS", "ECE", "CSE", "DS"],
        "Math": [85, 92, 45, 95, 60],
        "Physics": [90, 88, 52, 96, 65],
        "Chemistry": [92, 95, 48, 94, 58]
    }
    df_students = pd.DataFrame(data)

    # What is used: Calling add_calculated_columns.
    # Why it is used: Applies calculated columns and displays augmented table.
    # How it works: Prints augmented DataFrame.
    augmented_df = add_calculated_columns(df_students)
    print("--- Augmented Student DataFrame ---")
    print(augmented_df[["Name", "Total", "Average", "Result", "Grade"]])
