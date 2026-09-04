"""
Module: time_analysis.py
Performs time-series analytics, monthly aggregations, MoM growth rates (pct_change), and rolling averages.
"""

# What is used: Import pandas library.
# Why it is used: Core package for datetime grouping, lag differences, percentage changes, and rolling windows.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def analyze_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly revenue, cost, profit, order count, Month-over-Month (MoM) growth %, and 3-month rolling average.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Monthly time-series analysis summary.
    """
    res_df = df.copy()

    if "Order_Date" not in res_df.columns:
        raise KeyError("Order_Date column missing from DataFrame.")

    # What is used: df.dt.to_period("M").
    # Why it is used: Groups transactions into YYYY-MM period buckets chronologically.
    # How it works: Extracts year-month period string.
    res_df["Year_Month"] = res_df["Order_Date"].dt.to_period("M").astype(str)

    monthly = (
        res_df.groupby("Year_Month")
        .agg(
            total_revenue=("Revenue", "sum"),
            total_cost=("Cost", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order_ID", "count")
        )
        .reset_index()
    )

    monthly["total_revenue"] = monthly["total_revenue"].round(2)
    monthly["total_cost"] = monthly["total_cost"].round(2)
    monthly["total_profit"] = monthly["total_profit"].round(2)

    # What is used: shift(1), diff(), pct_change(), rolling(window=3).mean().
    # Why it is used: Calculates lag metrics, MoM growth percentages, and smoothed 3-month moving averages.
    # How it works: Computes differences and rolling window calculations across sequential monthly rows.
    monthly["Prev_Month_Revenue"] = monthly["total_revenue"].shift(1)
    monthly["Revenue_Diff"] = monthly["total_revenue"].diff().round(2)
    monthly["MoM_Growth_%"] = (monthly["total_revenue"].pct_change() * 100.0).round(2)
    monthly["Rolling_3M_Avg"] = monthly["total_revenue"].rolling(window=3).mean().round(2)

    return monthly


def analyze_daily_rolling_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily revenue and 7-day rolling average revenue.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Daily trend DataFrame with 7-day moving averages.
    """
    daily = (
        df.groupby("Order_Date")
        .agg(daily_revenue=("Revenue", "sum"), daily_orders=("Order_ID", "count"))
        .reset_index()
        .sort_values(by="Order_Date")
    )

    daily["daily_revenue"] = daily["daily_revenue"].round(2)

    # What is used: daily["daily_revenue"].rolling(window=7).mean().
    # Why it is used: Smoothes daily volatility with a 7-day moving window.
    # How it works: Computes average revenue over 7-day rolling windows.
    daily["7_Day_Rolling_Avg"] = daily["daily_revenue"].rolling(window=7).mean().round(2)

    return daily
