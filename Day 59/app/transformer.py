"""
Module: transformer.py
Computes derived financial metrics (Revenue, Cost, Profit) and date temporal components.
"""

# What is used: Import pandas library.
# Why it is used: Vectorized calculations across DataFrame columns.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich cleaned sales DataFrame with derived financial metrics and date components.

    Metrics Added:
    - Revenue: Quantity * Unit_Price * (1 - Discount)
    - Cost: Quantity * Cost_Price
    - Profit: Revenue - Cost
    - Profit_Margin_%: (Profit / Revenue) * 100
    - Year, Month, Month_Name, Day_Of_Week

    Args:
        df: Cleaned sales DataFrame.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    res_df = df.copy()

    # 1. Financial Metrics Calculation
    # What is used: Vectorized Series arithmetic.
    # Why it is used: Calculates total gross revenue, cost, profit, and margin percentage per order.
    # How it works: Applies mathematical formulas across Quantity, Unit_Price, Discount, and Cost_Price.
    res_df["Revenue"] = (
        res_df["Quantity"] * res_df["Unit_Price"] * (1.0 - res_df["Discount"])
    ).round(2)

    res_df["Cost"] = (res_df["Quantity"] * res_df["Cost_Price"]).round(2)
    res_df["Profit"] = (res_df["Revenue"] - res_df["Cost"]).round(2)
    res_df["Profit_Margin_%"] = (
        (res_df["Profit"] / res_df["Revenue"].replace(0, 1.0)) * 100.0
    ).round(2)

    # 2. Date Temporal Feature Extraction
    # What is used: Datetime accessor .dt.year, .dt.month, .dt.month_name(), .dt.day_name().
    # Why it is used: Extracts temporal components for downstream time-series analysis.
    # How it works: Extracts year, month number, month name, and day name from Order_Date.
    if "Order_Date" in res_df.columns:
        dt = res_df["Order_Date"].dt
        res_df["Year"] = dt.year
        res_df["Month"] = dt.month
        res_df["Month_Name"] = dt.month_name()
        res_df["Day_Of_Week"] = dt.day_name()

    return res_df
