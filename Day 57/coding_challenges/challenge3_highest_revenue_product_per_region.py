"""
Day 57 - Coding Challenge 3: Highest-Revenue Product inside each Region
Identify the product generating the highest total revenue inside each geographic region.
"""

# What is used: Import pandas library.
# Why it is used: Core package for multi-column grouping, sorting, and idxmax.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def get_top_product_per_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the product generating the maximum revenue inside each Region.

    Args:
        df: Input DataFrame containing Region, Product, and Revenue columns.

    Returns:
        pd.DataFrame: DataFrame containing Region, Product, and max Revenue.
    """
    # What is used: df.groupby(["Region", "Product"])["Revenue"].sum().reset_index().
    # Why it is used: Aggregates total revenue per Region and Product combination.
    # How it works: Sums revenue for each region-product pair.
    region_prod_rev = df.groupby(["Region", "Product"])["Revenue"].sum().reset_index()

    # What is used: region_prod_rev.sort_values(by=["Region", "Revenue"], ascending=[True, False]).groupby("Region").first().reset_index().
    # Why it is used: Sorts descending by revenue within each region and retains top product row per region.
    # How it works: Groups sorted DataFrame by Region and extracts first (highest revenue) row.
    top_per_region = (
        region_prod_rev.sort_values(by=["Region", "Revenue"], ascending=[True, False])
        .groupby("Region")
        .first()
        .reset_index()
    )

    return top_per_region


if __name__ == "__main__":
    # What is used: Dictionary of mock sales records across regions and products.
    # Why it is used: Serves as input dataset for per-region product analysis.
    # How it works: Maps Region, Product, Revenue fields.
    data = pd.DataFrame({
        "Region": ["West", "West", "East", "East", "South", "South"],
        "Product": ["Laptop", "Phone", "Phone", "Tablet", "Laptop", "Chair"],
        "Revenue": [80000, 60000, 50000, 20000, 90000, 15000]
    })

    # What is used: Calling get_top_product_per_region.
    # Why it is used: Extracts top product per region.
    # How it works: Prints top product per region table.
    top_products = get_top_product_per_region(data)
    print("--- Highest-Revenue Product per Region ---")
    print(top_products)
