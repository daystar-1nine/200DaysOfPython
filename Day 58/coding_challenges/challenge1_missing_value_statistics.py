"""
Day 58 - Coding Challenge 1: Missing Value Statistics Generator
Compute missing value counts and missing percentage Series for any input DataFrame.
"""

# What is used: Import pandas library.
# Why it is used: Core package for missing value detection and percentage computation.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def generate_missing_value_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate missing count and missing percentage statistics for a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table containing Missing_Count and Missing_Percentage.
    """
    # What is used: df.isna().sum().
    # Why it is used: Counts missing values per column.
    # How it works: Evaluates NaNs elementwise.
    missing_count = df.isna().sum()

    # What is used: (df.isna().mean() * 100.0).round(2).
    # Why it is used: Calculates percentage of missing values per column.
    # How it works: Computes proportion of True NaNs per column and multiplies by 100.
    missing_pct = (df.isna().mean() * 100.0).round(2)

    # What is used: pd.DataFrame constructor combining Series.
    # Why it is used: Constructs structured summary table.
    # How it works: Combines count and percentage Series into columns.
    stats_df = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_pct
    })

    return stats_df


if __name__ == "__main__":
    # What is used: Dictionary with missing values across multiple fields.
    # Why it is used: Test dataset for missing value statistics generator.
    # How it works: Holds None/NaN values in Age, Salary, City fields.
    data = pd.DataFrame({
        "Customer_ID": ["C01", "C02", "C03", "C04", "C05"],
        "Name": ["Rahul", "Priya", "Aman", "Sneha", "Vikram"],
        "Age": [20.0, None, 22.0, None, 25.0],
        "Salary": [50000.0, 60000.0, None, None, 75000.0],
        "City": ["Mumbai", None, "Delhi", "Pune", None]
    })

    # What is used: Calling generate_missing_value_stats.
    # Why it is used: Computes missing count and missing percentage per column.
    # How it works: Prints missing statistics table.
    missing_stats = generate_missing_value_stats(data)
    print("--- Missing Value Statistics ---")
    print(missing_stats)
