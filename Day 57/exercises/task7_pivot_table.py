"""
Day 57 - Practical Task 7: Reshaping with pivot_table()
Demonstrates creating a Region x Category sales pivot table with sum aggregation.
"""

# What is used: Import pandas library.
# Why it is used: Core package for pd.pivot_table operations.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def create_sales_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Region x Category sales pivot table summing Sales values.

    Args:
        df: Input DataFrame with Region, Category, and Sales columns.

    Returns:
        pd.DataFrame: Reshaped pivot table DataFrame.
    """
    # What is used: pd.pivot_table(df, values="Sales", index="Region", columns="Category", aggfunc="sum", fill_value=0).
    # Why it is used: Reshapes flat transactional data into a 2D matrix with summed sales per cell.
    # How it works: Groups by Region (rows) and Category (columns), aggregating Sales using sum function.
    pivot = pd.pivot_table(
        df,
        values="Sales",
        index="Region",
        columns="Category",
        aggfunc="sum",
        fill_value=0
    )
    return pivot


if __name__ == "__main__":
    # What is used: Dictionary defining e-commerce sales dataset.
    # Why it is used: Serves as test data for pivot table generation.
    # How it works: Contains Region, Category, and Sales values.
    sales_data = {
        "Region": ["West", "East", "West", "South", "East", "South"],
        "Category": ["Laptop", "Phone", "Phone", "Laptop", "Laptop", "Phone"],
        "Sales": [800, 500, 600, 900, 700, 400]
    }
    df_sales = pd.DataFrame(sales_data)

    # What is used: Calling create_sales_pivot_table.
    # Why it is used: Generates 2D Region x Category matrix.
    # How it works: Prints reshaped pivot table DataFrame.
    pivot_df = create_sales_pivot_table(df_sales)
    print("--- Region x Category Sales Pivot Table ---")
    print(pivot_df)
