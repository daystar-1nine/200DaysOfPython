"""
Module: overview.py
Computes macro-level business KPIs: total revenue, cost, profit, overall profit margin, total orders, and average order value (AOV).
"""

# What is used: Import pandas library.
# Why it is used: Aggregates top-level corporate sales figures.
# How it works: Calculates sums and means across financial columns.
import pandas as pd


def compute_overview_kpis(df: pd.DataFrame) -> dict:
    """
    Compute macro-level business performance indicators.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        dict: Top-level business KPIs.
    """
    total_orders = int(len(df))
    total_rev = round(float(df["Revenue"].sum()), 2)
    total_cost = round(float(df["Cost"].sum()), 2)
    total_profit = round(float(df["Profit"].sum()), 2)

    overall_margin = round((total_profit / total_rev * 100.0), 2) if total_rev > 0 else 0.0
    aov = round((total_rev / total_orders), 2) if total_orders > 0 else 0.0

    return {
        "total_orders": total_orders,
        "total_revenue": total_rev,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "overall_margin_%": overall_margin,
        "average_order_value": aov
    }
