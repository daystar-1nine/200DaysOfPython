"""
Day 58 - Practical Task 1: Missing Values Detection & Imputation
Demonstrates counting missing values, computing missing percentages, and imputing Age (median) and Marks (mean).
"""

# What is used: Import pandas and numpy libraries.
# Why it is used: Essential for NaN handling, missing statistics, and statistical imputation.
# How it works: Loads pandas and numpy packages into module scope.
import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Detect missing values, compute missing counts/percentages, and impute Age (median) and Marks (mean).

    Args:
        df: Input DataFrame with Age and Marks columns containing None/NaN.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: Imputed DataFrame, missing count Series, missing pct Series.
    """
    # What is used: df.isna().sum().
    # Why it is used: Counts missing values per column.
    # How it works: Evaluates boolean mask per element and sums True occurrences.
    missing_count = df.isna().sum()

    # What is used: (df.isna().mean() * 100).round(2).
    # Why it is used: Calculates percentage of missing values per column.
    # How it works: Computes proportion of True NaNs per column and multiplies by 100.
    missing_pct = (df.isna().mean() * 100.0).round(2)

    # What is used: df.copy() deep copy method.
    # Why it is used: Prevents mutating original input DataFrame parameter.
    # How it works: Duplicates memory buffer.
    imputed_df = df.copy()

    # What is used: Imputing Age with median and Marks with mean.
    # Why it is used: Median is resistant to outliers for Age; mean averages numeric Marks.
    # How it works: Calculates median/mean ignoring NaNs and replaces NaNs using fillna().
    age_median = float(imputed_df["Age"].median())
    marks_mean = float(imputed_df["Marks"].mean())

    imputed_df["Age"] = imputed_df["Age"].fillna(round(age_median, 1))
    imputed_df["Marks"] = imputed_df["Marks"].fillna(round(marks_mean, 2))

    return imputed_df, missing_count, missing_pct


if __name__ == "__main__":
    # What is used: Dictionary defining mock DataFrame with missing Age and Marks.
    # Why it is used: Test data for missing value detection and imputation.
    # How it works: Maps data containing None/NaN values.
    sample_data = {
        "Name": ["A", "B", "C", "D"],
        "Age": [20.0, None, 22.0, None],
        "Marks": [80.0, 90.0, None, 70.0]
    }
    df = pd.DataFrame(sample_data)

    # What is used: Calling handle_missing_values function.
    # Why it is used: Performs missing value analysis and imputation.
    # How it works: Displays missing statistics and imputed DataFrame.
    imputed_df, counts, pcts = handle_missing_values(df)

    print("--- Missing Count per Column ---")
    print(counts)

    print("\n--- Missing Percentage per Column (%) ---")
    print(pcts)

    print("\n--- Imputed DataFrame ---")
    print(imputed_df)
