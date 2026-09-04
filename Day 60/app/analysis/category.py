"""
Module: category.py
Analyzes product category revenue, gross profit, sales quantity, and average discount levels.
"""

# What is used: Import pandas library.
# Why it is used: Grouped aggregations by Category dimension.
# How it works: Aggregates revenue, profit, quantity, and discount per category and computes ranks.
import pandas as pd


def analyze_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform category-level performance and profitability analysis.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Category summary table sorted by total revenue descending.
    """
    res = (
        df.groupby("Category")
        .agg(
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            total_quantity=("Quantity", "sum"),
            order_count=("Order_ID", "count"),
            avg_discount=("Discount", "mean"),
            profit_margin=("Profit_Margin", "mean")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["avg_discount_%"] = (res["avg_discount"] * 100.0).round(2)
    res["profit_margin_%"] = res["profit_margin"].round(2)
    res = res.drop(columns=["avg_discount", "profit_margin"])

    total_sales = res["total_revenue"].sum()
    res["Revenue_Share_%"] = ((res["total_revenue"] / total_sales) * 100.0).round(2)
    res["Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    return res.sort_values(by="Rank").reset_index(drop=True)
