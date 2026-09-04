"""
Module: product.py
Analyzes product performance metrics, revenue ranks, and intra-category rankings.
"""

# What is used: Import pandas library.
# Why it is used: Grouped aggregations by Category and Product.
# How it works: Aggregates revenue, profit, and quantity per product and ranks them.
import pandas as pd


def analyze_products(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute product performance summaries, overall ranks, and intra-category ranks.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        (full_product_summary, top_by_revenue, top_by_quantity, top_by_profit)
    """
    res = (
        df.groupby(["Category", "Product"])
        .agg(
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            total_quantity=("Quantity", "sum"),
            order_count=("Order_ID", "count"),
            avg_unit_price=("Unit_Price", "mean")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["avg_unit_price"] = res["avg_unit_price"].round(2)

    # What is used: groupby().rank() for category partitions and overall ranks.
    # Why it is used: Ranks products within each category and across the entire catalog.
    # How it works: Computes dense ranks descending.
    res["Category_Rank"] = res.groupby("Category")["total_revenue"].rank(ascending=False, method="dense").astype(int)
    res["Overall_Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    top_revenue = res.sort_values(by="total_revenue", ascending=False).head(10).reset_index(drop=True)
    top_quantity = res.sort_values(by="total_quantity", ascending=False).head(10).reset_index(drop=True)
    top_profit = res.sort_values(by="total_profit", ascending=False).head(10).reset_index(drop=True)

    return res.sort_values(by="Overall_Rank").reset_index(drop=True), top_revenue, top_quantity, top_profit
