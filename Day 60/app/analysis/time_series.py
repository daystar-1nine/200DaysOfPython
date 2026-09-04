"""
Module: time_series.py
Performs temporal trend analytics, monthly aggregations, MoM growth calculations, and moving averages.
"""

# What is used: Import pandas library.
# Why it is used: Temporal aggregations, shift(), pct_change(), and rolling() windows.
# How it works: Aggregates data by YYYY-MM and date, and applies lag and window functions.
import pandas as pd


def analyze_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly revenue, cost, profit, order count, MoM growth %, and 3-month rolling averages.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Monthly time-series DataFrame.
    """
    res_df = df.copy()
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

    # What is used: shift(1), pct_change(), rolling(3).mean().
    # Why it is used: Computes lag revenue, growth percentage, and 3-month smoothed moving averages.
    # How it works: Applies time-series operations over chronological monthly rows.
    monthly["Prev_Month_Revenue"] = monthly["total_revenue"].shift(1)
    monthly["MoM_Growth_%"] = (monthly["total_revenue"].pct_change() * 100.0).round(2)
    monthly["Rolling_3M_Avg"] = monthly["total_revenue"].rolling(window=3).mean().round(2)

    return monthly


def analyze_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily revenue and 7-day rolling average revenue.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Daily trend DataFrame with 7-day moving averages.
    """
    daily = (
        df.groupby("Order_Date")
        .agg(daily_revenue=("Revenue", "sum"), order_count=("Order_ID", "count"))
        .reset_index()
        .sort_values(by="Order_Date")
    )

    daily["daily_revenue"] = daily["daily_revenue"].round(2)
    daily["7_Day_Rolling_Avg"] = daily["daily_revenue"].rolling(window=7).mean().round(2)

    return daily
