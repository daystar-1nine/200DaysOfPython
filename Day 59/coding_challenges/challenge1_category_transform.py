"""
Day 59 - Coding Challenge 1: Product Performance vs Category Average via transform()
Calculates category mean sales, sales differences, and flags above-average products.
"""

# What is used: Import sys and pandas modules.
# Why it is used: Configures UTF-8 console output and performs group transform calculations.
# How it works: Brings sys and pandas namespaces into scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compare_sales_to_category_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute row-aligned category average sales, difference, and boolean performance flag.

    Args:
        df: Input DataFrame containing 'Category' and 'Sales' columns.

    Returns:
        pd.DataFrame: Enriched DataFrame with Category_Avg, Difference, and Above_Average columns.
    """
    result_df = df.copy()

    # What is used: groupby("Category")["Sales"].transform("mean").
    # Why it is used: Broadcasts category mean across original rows without changing DataFrame length.
    # How it works: Computes average per category and assigns to aligned Series.
    result_df["Category_Avg"] = (
        result_df.groupby("Category")["Sales"]
        .transform("mean")
        .round(2)
    )

    # What is used: Vectorized arithmetic and comparison.
    # Why it is used: Computes relative difference and boolean above-average indicator.
    # How it works: Subtracts Category_Avg from Sales and evaluates Sales > Category_Avg.
    result_df["Difference"] = (result_df["Sales"] - result_df["Category_Avg"]).round(2)
    result_df["Above_Average"] = result_df["Sales"] > result_df["Category_Avg"]

    return result_df


def main() -> None:
    data = {
        "Product": ["Laptop Pro", "Laptop Air", "Phone X", "Phone Y", "Tablet A", "Tablet B"],
        "Category": ["Electronics", "Electronics", "Electronics", "Electronics", "Gadgets", "Gadgets"],
        "Sales": [85000, 62000, 45000, 52000, 30000, 42000]
    }
    df = pd.DataFrame(data)
    processed = compare_sales_to_category_average(df)
    print("--- Product Performance vs Category Average ---")
    print(processed)


if __name__ == "__main__":
    main()
