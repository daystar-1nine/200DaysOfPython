"""
Day 57 - Practical Task 5: Function Mapping with apply()
Demonstrates applying a custom grading function to a Marks Series using apply().
"""

# What is used: Import pandas library.
# Why it is used: Core library for Series.apply operations.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def get_grade(mark: float) -> str:
    """
    Determine letter grade based on percentage mark.

    Args:
        mark: Numerical score.

    Returns:
        str: Letter grade (A+, A, B, C, F).
    """
    if mark >= 90:
        return "A+"
    elif mark >= 80:
        return "A"
    elif mark >= 70:
        return "B"
    elif mark >= 60:
        return "C"
    else:
        return "F"


def assign_grades_with_apply(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply get_grade function to Marks column to populate Grade column.

    Args:
        df: Input DataFrame containing Marks column.

    Returns:
        pd.DataFrame: Augmented DataFrame with Grade column.
    """
    # What is used: df.copy() deep copy method.
    # Why it is used: Ensures input DataFrame is not mutated.
    # How it works: Creates independent DataFrame instance.
    res_df = df.copy()

    # What is used: res_df["Marks"].apply(get_grade).
    # Why it is used: Evaluates get_grade custom function for every element in the Marks Series.
    # How it works: Passes each mark scalar to get_grade and assigns return string to Grade column.
    res_df["Grade"] = res_df["Marks"].apply(get_grade)
    return res_df


if __name__ == "__main__":
    # What is used: Dictionary of student marks.
    # Why it is used: Test data for apply() grade assignment.
    # How it works: Maps student names to scores.
    data = {
        "Name": ["Aarav", "Kabir", "Rohan", "Tanvi", "Vikram"],
        "Marks": [95, 45, 82, 88, 65]
    }
    df_students = pd.DataFrame(data)

    # What is used: Calling assign_grades_with_apply.
    # Why it is used: Applies get_grade function across Marks Series.
    # How it works: Displays output table with Grade column.
    graded_df = assign_grades_with_apply(df_students)
    print("--- Student Grade Assignments (apply) ---")
    print(graded_df)
