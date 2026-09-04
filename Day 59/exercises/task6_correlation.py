"""
Day 59 - Exercise 6: Correlation Analysis & Relationship Interpretation
Demonstrates numerical correlation matrix computation using df.corr(numeric_only=True).
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and computes Pearson correlation matrix.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # What is used: Sample student metrics dataset creation.
    # Why it is used: Provides numerical features to measure linear relationships.
    # How it works: Initializes DataFrame with Study_Hours, Attendance, and Marks.
    data = {
        "Study_Hours": [2.5, 5.0, 7.5, 1.5, 9.0, 6.0, 3.5, 8.0],
        "Attendance": [70, 85, 90, 60, 98, 88, 75, 95],
        "Marks": [55, 72, 88, 45, 96, 80, 62, 91]
    }
    df = pd.DataFrame(data)

    # What is used: df.corr(numeric_only=True).
    # Why it is used: Computes pairwise Pearson correlation coefficients between numerical columns.
    # How it works: Returns symmetric 3x3 correlation matrix with values between -1 and +1.
    corr_matrix = df.corr(numeric_only=True).round(4)

    print("--- Student Metrics Correlation Matrix ---")
    print(corr_matrix)
    print("\n--- Relationship Insights ---")
    print(f"Study_Hours vs Marks Correlation: {corr_matrix.loc['Study_Hours', 'Marks']} (Strong Positive)")
    print(f"Attendance vs Marks Correlation : {corr_matrix.loc['Attendance', 'Marks']} (Strong Positive)")


if __name__ == "__main__":
    main()
