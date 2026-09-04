"""
Day 57 - Practical Task 3: Relational DataFrame Merging
Demonstrates pd.merge() to join employee data with department details.
"""

# What is used: Import pandas library.
# Why it is used: Core package for relational merging operations.
# How it works: Brings pandas package into namespace.
import pandas as pd


def merge_employees_and_departments(emp_df: pd.DataFrame, dept_df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """
    Merge employee DataFrame with department DataFrame on Dept_ID key.

    Args:
        emp_df: DataFrame containing employee details and Dept_ID.
        dept_df: DataFrame containing Dept_ID and Dept_Name details.
        how: Merge type ('inner', 'left', 'right', 'outer').

    Returns:
        pd.DataFrame: Joined DataFrame.
    """
    # What is used: pd.merge(emp_df, dept_df, on="Dept_ID", how=how).
    # Why it is used: Performs relational SQL-style JOIN between two DataFrames on join key.
    # How it works: Matches row entries where Dept_ID values align according to specified join mode.
    merged = pd.merge(emp_df, dept_df, on="Dept_ID", how=how)
    return merged


if __name__ == "__main__":
    # What is used: Dictionaries for employees and departments.
    # Why it is used: Provides mock relational tables to test merge.
    # How it works: Maps employee and department entities.
    employees_data = {
        "Emp_ID": [101, 102, 103, 104],
        "Emp_Name": ["Rahul", "Priya", "Aman", "Sneha"],
        "Dept_ID": [1, 2, 1, 3]
    }
    departments_data = {
        "Dept_ID": [1, 2, 3, 4],
        "Dept_Name": ["Engineering", "Data Science", "Hardware", "Marketing"]
    }

    # What is used: pd.DataFrame construction.
    # Why it is used: Builds two relational DataFrames.
    # How it works: Converts dictionaries to tables.
    emp_df = pd.DataFrame(employees_data)
    dept_df = pd.DataFrame(departments_data)

    # What is used: Calling merge_employees_and_departments.
    # Why it is used: Executes inner join merge.
    # How it works: Displays joined DataFrame containing employee names and department names.
    merged_df = merge_employees_and_departments(emp_df, dept_df, how="inner")
    print("--- Inner Merged Employee Department Data ---")
    print(merged_df)
