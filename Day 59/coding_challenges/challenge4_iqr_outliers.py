"""
Day 59 - Coding Challenge 4: Modular IQR Outlier Detection Engine
Calculates Q1, Q3, IQR, lower/upper boundaries, outlier counts, and outlier rows for any numeric column.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and computes IQR outlier statistics.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def detect_column_outliers(df: pd.DataFrame, col_name: str) -> dict:
    """
    Detect statistical outliers in a specified numerical column using the IQR method.

    Args:
        df: Input DataFrame.
        col_name: Name of numerical column to analyze.

    Returns:
        dict: Outlier metrics dictionary containing q1, q3, iqr, bounds, count, and outlier DataFrame.
    """
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' does not exist in DataFrame.")

    series = df[col_name].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
    outlier_df = df[outlier_mask].copy()

    return {
        "column": col_name,
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": len(outlier_df),
        "outliers": outlier_df
    }


def main() -> None:
    data = {
        "Order_ID": range(101, 116),
        "Amount": [200, 220, 210, 190, 205, 215, 225, 200, 195, 210, 205, 230, 15, 1200, 1500]
    }
    df = pd.DataFrame(data)
    results = detect_column_outliers(df, "Amount")

    print(f"--- Outlier Audit for Column: {results['column']} ---")
    print(f"Q1: {results['q1']} | Q3: {results['q3']} | IQR: {results['iqr']}")
    print(f"Lower Bound: {results['lower_bound']} | Upper Bound: {results['upper_bound']}")
    print(f"Outlier Count: {results['outlier_count']}")
    print("\n--- Outlier Rows ---")
    print(results["outliers"])


if __name__ == "__main__":
    main()
