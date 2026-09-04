"""
Module: customer.py
Analyzes customer revenue contributions, order counts, average order values, and ranking distributions.
"""

# What is used: Import pandas library.
# Why it is used: Grouped aggregations by Customer_ID and Customer_Name.
# How it works: Groups transactions per customer, aggregates financials, and computes customer ranks.
import pandas as pd


def analyze_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform customer-level analytics and generate customer revenue rankings.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Customer summary DataFrame sorted by total revenue descending.
    """
    res = (
        df.groupby(["Customer_ID", "Customer_Name"])
        .agg(
            total_revenue=("Revenue", "sum"),
            total_cost=("Cost", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order_ID", "count"),
            aov=("Revenue", "mean")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_cost"] = res["total_cost"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["aov"] = res["aov"].round(2)

    res["Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    avg_spend = res["total_revenue"].mean()
    res["Above_Average_Spend"] = res["total_revenue"] > avg_spend

    return res.sort_values(by="Rank").reset_index(drop=True)
