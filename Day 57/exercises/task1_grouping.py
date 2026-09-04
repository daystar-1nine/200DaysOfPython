"""
Day 57 - Practical Task 1: Basic Groupby Aggregations
Demonstrates grouping by Department and computing Total, Average, Max, and Min salary.
"""

# What is used: Import pandas library.
# Why it is used: Essential for groupby aggregation operations.
# How it works: Loads pandas package into module namespace.
import pandas as pd


def analyze_department_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group employees by Department and calculate total, mean, min, and max salary.

    Args:
        df: Input DataFrame with Department and Salary columns.

    Returns:
        pd.DataFrame: Aggregated summary table.
    """
    # What is used: df.groupby("Department")["Salary"].agg() with named aggregations.
    # Why it is used: Computes multiple descriptive statistics per department cleanly.
    # How it works: Groups rows by Department and evaluates sum, mean, min, max on Salary Series.
    result = df.groupby("Department").agg(
        total_salary=("Salary", "sum"),
        average_salary=("Salary", "mean"),
        max_salary=("Salary", "max"),
        min_salary=("Salary", "min")
    ).reset_index()

    return result


if __name__ == "__main__":
    # What is used: Dictionary defining employee salary data.
    # Why it is used: Serves as test dataset for department grouping.
    # How it works: Maps column fields to value lists.
    sample_data = {
        "Department": ["CSE", "CSE", "DS", "DS", "ECE", "ECE"],
        "Salary": [50000, 60000, 70000, 80000, 55000, 65000]
    }

    # What is used: pd.DataFrame constructor.
    # Why it is used: Instantiates 2D DataFrame from input dictionary.
    # How it works: Aligns dictionary keys into columns.
    df = pd.DataFrame(sample_data)

    # What is used: Calling analyze_department_salaries.
    # Why it is used: Runs groupby aggregations and prints output.
    # How it works: Displays aggregated salary metrics.
    dept_summary = analyze_department_salaries(df)
    print("--- Department Salary Summary ---")
    print(dept_summary)
