"""
Day 57 - Practical Task 2: Multi-Column Groupby
Demonstrates grouping by Department and Gender to compute average salary.
"""

# What is used: Import pandas library.
# Why it is used: Core package for multi-column grouping.
# How it works: Brings pandas namespace into execution context.
import pandas as pd


def analyze_dept_gender_salary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average salary grouped by Department and Gender.

    Args:
        df: Input DataFrame containing Department, Gender, and Salary columns.

    Returns:
        pd.DataFrame: DataFrame with average salary grouped by Department and Gender.
    """
    # What is used: df.groupby(["Department", "Gender"])["Salary"].mean().reset_index().
    # Why it is used: Performs two-level hierarchical grouping and converts Series back to DataFrame.
    # How it works: Splits table into Dept+Gender subgroups and calculates mean salary.
    result = df.groupby(["Department", "Gender"])["Salary"].mean().reset_index()
    result.rename(columns={"Salary": "Average_Salary"}, inplace=True)
    return result


if __name__ == "__main__":
    # What is used: Dictionary defining mock employee dataset with Department, Gender, Salary.
    # Why it is used: Provides multi-categorical data for multi-column grouping test.
    # How it works: Maps data lists into dictionary keys.
    data = {
        "Department": ["CSE", "CSE", "DS", "DS", "ECE", "ECE"],
        "Gender": ["M", "F", "M", "F", "M", "F"],
        "Salary": [55000, 62000, 75000, 81000, 58000, 64000]
    }

    # What is used: pd.DataFrame constructor.
    # Why it is used: Creates Pandas DataFrame from dictionary.
    # How it works: Builds 2D table object.
    df = pd.DataFrame(data)

    # What is used: Calling analyze_dept_gender_salary.
    # Why it is used: Executes multi-column grouping.
    # How it works: Displays average salary broken down by Department and Gender.
    summary = analyze_dept_gender_salary(df)
    print("--- Department & Gender Average Salary ---")
    print(summary)
