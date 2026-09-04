"""
Module: transformer.py
Computes derived financial metrics (Revenue, Cost, Profit, Profit Margin %) and extracts temporal features.
"""

# What is used: Import pandas library.
# Why it is used: Vectorized feature calculations across DataFrame columns.
# How it works: Applies formulas for revenue, cost, profit, and margin, and extracts date parts.
import pandas as pd


def transform_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and enrich sales dataset with financial KPIs and date features.

    Metrics Computed:
    - Revenue: Quantity * Unit_Price * (1 - Discount)
    - Cost: Quantity * Cost_Price
    - Profit: Revenue - Cost
    - Profit_Margin: (Profit / Revenue) * 100 (safely handling Revenue == 0)
    - Year, Month, Month_Name, Day_Of_Week

    Args:
        df: Cleaned sales DataFrame.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    res = df.copy()

    # 1. Financial Metrics
    res["Revenue"] = (res["Quantity"] * res["Unit_Price"] * (1.0 - res["Discount"])).round(2)
    res["Cost"] = (res["Quantity"] * res["Cost_Price"]).round(2)
    res["Profit"] = (res["Revenue"] - res["Cost"]).round(2)

    # Safe Profit Margin computation (avoid divide-by-zero)
    safe_rev = res["Revenue"].replace(0, 1.0)
    res["Profit_Margin"] = ((res["Profit"] / safe_rev) * 100.0).round(2)
    res.loc[res["Revenue"] == 0, "Profit_Margin"] = 0.0

    # 2. Date Components
    if "Order_Date" in res.columns:
        dt = res["Order_Date"].dt
        res["Year"] = dt.year
        res["Month"] = dt.month
        res["Month_Name"] = dt.month_name()
        res["Day_Of_Week"] = dt.day_name()

    return res
