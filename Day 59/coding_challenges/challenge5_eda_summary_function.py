"""
Day 59 - Coding Challenge 5: Reusable Automated EDA Summary Function
Implements generate_eda_summary(df) returning shape, missing, duplicates, numerical summary, correlations, and outliers.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and computes automated EDA metrics.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_eda_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive automated Exploratory Data Analysis summary for any DataFrame.

    Args:
        df: Input DataFrame to profile.

    Returns:
        dict: Summary dictionary containing shape, missing counts, duplicates, descriptive stats, correlations, and IQR outlier counts.
    """
    summary = {}

    # 1. Dataset Shape & Meta
    summary["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

    # 2. Missing Values & Duplicates
    summary["missing_values"] = df.isna().sum().to_dict()
    summary["duplicate_rows"] = int(df.duplicated().sum())

    # 3. Numerical Summary Statistics
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        summary["numerical_summary"] = num_df.describe().round(2).to_dict()
        summary["correlation_matrix"] = num_df.corr().round(4).to_dict()

        # 4. Outlier Counts per Numeric Column
        outlier_counts = {}
        for col in num_df.columns:
            q1 = num_df[col].quantile(0.25)
            q3 = num_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            c = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())
            outlier_counts[col] = c
        summary["outlier_counts"] = outlier_counts

    # 5. Categorical Class Counts
    cat_df = df.select_dtypes(include=["object", "category"])
    cat_summary = {}
    for col in cat_df.columns:
        cat_summary[col] = cat_df[col].value_counts().head(5).to_dict()
    summary["categorical_top5"] = cat_summary

    return summary


def main() -> None:
    data = {
        "Customer_ID": [f"C10{i}" for i in range(1, 11)],
        "Age": [25, 30, 28, 45, 120, 22, 35, 29, 31, -5],
        "Salary": [50000, 60000, 55000, 80000, 500000, 45000, 70000, 58000, 62000, 1000],
        "City": ["Mumbai", "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai", "Pune"]
    }
    df = pd.DataFrame(data)
    eda = generate_eda_summary(df)

    print("==================================================")
    print("           AUTOMATED EDA SUMMARY REPORT           ")
    print("==================================================")
    print(f"Shape            : {eda['shape']}")
    print(f"Duplicates       : {eda['duplicate_rows']}")
    print(f"Missing Values   : {eda['missing_values']}")
    print(f"Outlier Counts   : {eda.get('outlier_counts', {})}")
    print(f"Categorical Top 5: {eda['categorical_top5']}")


if __name__ == "__main__":
    main()
