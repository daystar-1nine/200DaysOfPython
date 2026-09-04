"""
Day 59 - Exercise 5: IQR Outlier Detection on Salary Dataset
Demonstrates calculation of Q1, Q3, IQR, and lower/upper outlier boundary filtering.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and performs quantile computations.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # What is used: Sample salary dataset containing extreme outlier values.
    # Why it is used: Demonstrates statistical outlier boundary identification.
    # How it works: Constructs DataFrame with standard salaries and extreme outliers (5,000 and 500,000).
    salaries = [
        45000, 48000, 52000, 50000, 55000, 58000, 60000, 62000,
        51000, 49000, 53000, 56000, 5000, 500000
    ]
    df = pd.DataFrame({"Salary": salaries})

    # What is used: df["Salary"].quantile(0.25) and quantile(0.75).
    # Why it is used: Calculates 25th percentile (Q1) and 75th percentile (Q3).
    # How it works: Computes quartiles and derives Interquartile Range (IQR).
    q1 = df["Salary"].quantile(0.25)
    q3 = df["Salary"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # What is used: Boolean mask filtering.
    # Why it is used: Identifies rows exceeding lower and upper IQR boundaries.
    # How it works: Selects records where Salary < lower_bound or Salary > upper_bound.
    outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]

    print(f"Q1 (25th Percentile) : {q1}")
    print(f"Q3 (75th Percentile) : {q3}")
    print(f"IQR                  : {iqr}")
    print(f"Lower Bound          : {lower_bound}")
    print(f"Upper Bound          : {upper_bound}")
    print("\n--- Detected Outliers ---")
    print(outliers)


if __name__ == "__main__":
    main()
