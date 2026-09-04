"""
Day 56 - Practical Task 3: Filtering & Boolean Masking
Demonstrates multi-condition boolean filtering on DataFrames using bitwise operators (&, |, ~).
"""

# What is used: Import pandas module.
# Why it is used: Core package for data filtering.
# How it works: Imports pandas namespace.
import pandas as pd


def perform_filtering_practice(df: pd.DataFrame) -> dict:
    """
    Perform various multi-condition filtering queries on student dataset.

    Args:
        df: Input student DataFrame.

    Returns:
        dict: Filtered DataFrames keyed by filter name.
    """
    # What is used: Bitwise AND filtering (df["Department"] == "CSE") & (df["Math"] >= 90).
    # Why it is used: Filters students in CSE department scoring 90+ in Math.
    # How it works: Evaluates two boolean Series elementwise.
    cse_high_math = df[(df["Department"] == "CSE") & (df["Math"] >= 90)].copy()

    # What is used: Bitwise OR filtering (df["Math"] >= 90) | (df["Physics"] >= 90).
    # Why it is used: Identifies students who excelled in either Math OR Physics.
    # How it works: Combines boolean vectors returning True if either condition holds.
    star_performers = df[(df["Math"] >= 90) | (df["Physics"] >= 90)].copy()

    # What is used: Bitwise NOT (~) and isin() filtering ~df["Department"].isin(["ECE"]).
    # Why it is used: Excludes students in the ECE department.
    # How it works: Inverts boolean mask returned by isin(["ECE"]).
    non_ece = df[~df["Department"].isin(["ECE"])].copy()

    return {
        "cse_high_math": cse_high_math,
        "star_performers": star_performers,
        "non_ece": non_ece
    }


if __name__ == "__main__":
    # What is used: Dictionary defining sample student records.
    # Why it is used: Input dataset for testing filtering conditions.
    # How it works: Maps student attributes into dictionary.
    data = {
        "Student_ID": ["S101", "S102", "S103", "S104", "S105"],
        "Name": ["Aarav", "Ananya", "Rohan", "Priya", "Vikram"],
        "Department": ["CSE", "DS", "ECE", "CSE", "DS"],
        "Math": [85, 92, 78, 95, 60],
        "Physics": [90, 88, 82, 96, 65],
        "Chemistry": [92, 95, 80, 94, 58]
    }
    df_students = pd.DataFrame(data)

    # What is used: Calling perform_filtering_practice function.
    # Why it is used: Executes filtering logic and prints results.
    # How it works: Displays filtered views of the student table.
    filtered_results = perform_filtering_practice(df_students)

    print("--- CSE Students with Math >= 90 ---")
    print(filtered_results["cse_high_math"][["Name", "Department", "Math"]])

    print("\n--- Star Performers (Math >= 90 OR Physics >= 90) ---")
    print(filtered_results["star_performers"][["Name", "Math", "Physics"]])

    print("\n--- Non-ECE Students ---")
    print(filtered_results["non_ece"][["Name", "Department"]])
