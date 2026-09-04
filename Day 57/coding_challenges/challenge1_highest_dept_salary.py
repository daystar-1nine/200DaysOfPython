"""
Day 57 - Coding Challenge 1: Department with Highest Average Salary
Find the department with the highest average salary from employee DataFrame.
"""

# What is used: Import pandas library.
# Why it is used: Core package for groupby aggregation and sorting.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def get_highest_avg_salary_dept(df: pd.DataFrame) -> tuple[str, float]:
    """
    Identify the department with the highest average salary.

    Args:
        df: Input DataFrame containing Department and Salary columns.

    Returns:
        tuple[str, float]: Department name and highest average salary value.
    """
    # What is used: df.groupby("Department")["Salary"].mean().
    # Why it is used: Computes mean salary for each department.
    # How it works: Returns pandas Series indexed by Department.
    avg_salaries = df.groupby("Department")["Salary"].mean()

    # What is used: avg_salaries.idxmax() and avg_salaries.max().
    # Why it is used: Finds index key of maximum value and maximum scalar value itself.
    # How it works: Scans Series elements to identify peak average salary department.
    best_dept = str(avg_salaries.idxmax())
    highest_avg = float(avg_salaries.max())

    return best_dept, highest_avg


if __name__ == "__main__":
    # What is used: Dictionary defining test dataset.
    # Why it is used: Provides mock employee data.
    # How it works: Maps data columns to lists.
    df = pd.DataFrame({
        "Department": ["CSE", "CSE", "DS", "DS", "ECE"],
        "Salary": [50000, 60000, 70000, 80000, 55000]
    })

    # What is used: Calling get_highest_avg_salary_dept.
    # Why it is used: Determines highest average salary department.
    # How it works: Prints department name and average salary.
    dept, max_avg = get_highest_avg_salary_dept(df)
    print(f"Highest Average Salary Department: {dept} (${max_avg:,.2f})")
