"""
Day 59 - Exercise 1: Department Average & Difference using transform()
Demonstrates broadcast row alignment using groupby().transform("mean") without reducing DataFrame size.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and handles DataFrame operations.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # What is used: Sample employee DataFrame creation.
    # Why it is used: Provides dummy department and salary data for broadcast testing.
    # How it works: Initializes DataFrame with Department and Salary columns.
    data = {
        "Employee": ["Rahul", "Priya", "Aman", "Sneha", "Vikram", "Ananya"],
        "Department": ["Engineering", "Engineering", "Marketing", "Engineering", "Marketing", "Marketing"],
        "Salary": [80000, 60000, 45000, 95000, 50000, 55000]
    }
    df = pd.DataFrame(data)

    # What is used: df.groupby("Department")["Salary"].transform("mean").
    # Why it is used: Broadcasts department average salary to every matching row.
    # How it works: Calculates mean per department and returns aligned Series.
    df["Department_Average"] = df.groupby("Department")["Salary"].transform("mean")

    # What is used: Vectorized arithmetic subtraction.
    # Why it is used: Computes difference between employee salary and department average.
    # How it works: Subtracts Department_Average from Salary per row.
    df["Difference_From_Average"] = df["Salary"] - df["Department_Average"]

    print("--- Employee Salary vs Department Average ---")
    print(df)


if __name__ == "__main__":
    main()
