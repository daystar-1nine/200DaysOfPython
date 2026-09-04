"""
Day 57 - Coding Challenge 4: Region x Product Revenue Pivot Table
Generate a 2D pivot table showing total revenue across Region (rows) and Product (columns).
"""

# What is used: Import pandas library.
# Why it is used: Core package for pd.pivot_table operations.
# How it works: Imports pandas namespace.
import pandas as pd


def create_region_product_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Region x Product pivot table with total revenue values.

    Args:
        df: Input DataFrame containing Region, Product, and Revenue columns.

    Returns:
        pd.DataFrame: Reshaped pivot table DataFrame.
    """
    # What is used: pd.pivot_table(df, values="Revenue", index="Region", columns="Product", aggfunc="sum", fill_value=0).
    # Why it is used: Transforms flat tabular data into 2D Region vs Product grid matrix.
    # How it works: Groups by Region index and Product columns, summing Revenue and filling missing cells with 0.
    pivot = pd.pivot_table(
        df,
        values="Revenue",
        index="Region",
        columns="Product",
        aggfunc="sum",
        fill_value=0
    )
    return pivot


if __name__ == "__main__":
    # What is used: Mock sales dataset with Region, Product, Revenue.
    # Why it is used: Provides data to construct Region x Product pivot grid.
    # How it works: Maps sales records into DataFrame.
    data = pd.DataFrame({
        "Region": ["West", "East", "West", "South", "East", "South"],
        "Product": ["Laptop", "Phone", "Phone", "Laptop", "Laptop", "Phone"],
        "Revenue": [80000, 50000, 60000, 90000, 70000, 40000]
    })

    # What is used: Calling create_region_product_pivot.
    # Why it is used: Generates pivot table view.
    # How it works: Prints reshaped Region x Product pivot matrix.
    pivot_table_df = create_region_product_pivot(data)
    print("--- Region x Product Revenue Pivot Table ---")
    print(pivot_table_df)
