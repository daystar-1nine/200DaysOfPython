"""
Day 56 - Practical Task 4: Sorting & Ranking Practice
Demonstrates single and multi-column DataFrame sorting using df.sort_values().
"""

# What is used: Import pandas library.
# Why it is used: Core library for DataFrame sorting operations.
# How it works: Imports pandas namespace.
import pandas as pd


def perform_sorting_practice(df: pd.DataFrame) -> dict:
    """
    Perform single and multi-column sorting operations on student DataFrame.

    Args:
        df: Input DataFrame containing Math and Total columns.

    Returns:
        dict: Collection of sorted DataFrames.
    """
    # What is used: df.sort_values(by="Math", ascending=False).
    # Why it is used: Sorts student records by Math score in descending order.
    # How it works: Orders DataFrame rows based on Math Series values.
    sorted_math = df.sort_values(by="Math", ascending=False).copy()

    # What is used: Multi-column sorting df.sort_values(by=["Department", "Math"], ascending=[True, False]).
    # Why it is used: Sorts primary by Department alphabetically, then secondary by Math score descending.
    # How it works: Evaluates department groups first, then sorts within each department group.
    sorted_dept_math = df.sort_values(by=["Department", "Math"], ascending=[True, False]).copy()

    return {
        "sorted_math": sorted_math,
        "sorted_dept_math": sorted_dept_math
    }


if __name__ == "__main__":
    # What is used: Mock dictionary of student attributes.
    # Why it is used: Test data for sorting logic.
    # How it works: Converts raw data dictionary to DataFrame.
    data = {
        "Student_ID": ["S101", "S102", "S103", "S104", "S105"],
        "Name": ["Aarav", "Ananya", "Rohan", "Priya", "Vikram"],
        "Department": ["CSE", "DS", "ECE", "CSE", "DS"],
        "Math": [85, 92, 78, 95, 60]
    }
    df_students = pd.DataFrame(data)

    # What is used: Calling perform_sorting_practice.
    # Why it is used: Executes sorting routines and displays sorted tables.
    # How it works: Prints sorted outputs.
    results = perform_sorting_practice(df_students)

    print("--- Sorted by Math Score (Descending) ---")
    print(results["sorted_math"][["Name", "Department", "Math"]])

    print("\n--- Sorted by Department (Asc) and Math Score (Desc) ---")
    print(results["sorted_dept_math"][["Department", "Name", "Math"]])
