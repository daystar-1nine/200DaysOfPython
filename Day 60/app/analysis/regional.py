"""
Module: regional.py
Analyzes regional sales performance, profit contributions, and order volumes.
"""

# What is used: Import pandas library.
# Why it is used: Grouped aggregations by Region dimension.
# How it works: Aggregates revenue, cost, profit, and order count, and computes regional ranks.
import pandas as pd


def analyze_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform regional sales performance analysis.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Regional performance table sorted by total revenue descending.
    """
    res = (
        df.groupby("Region")
        .agg(
            total_revenue=("Revenue", "sum"),
            total_cost=("Cost", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order_ID", "count"),
            average_order_revenue=("Revenue", "mean")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_cost"] = res["total_cost"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["average_order_revenue"] = res["average_order_revenue"].round(2)

    total_sales = res["total_revenue"].sum()
    res["Revenue_Share_%"] = ((res["total_revenue"] / total_sales) * 100.0).round(2)
    res["Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    return res.sort_values(by="Rank").reset_index(drop=True)
