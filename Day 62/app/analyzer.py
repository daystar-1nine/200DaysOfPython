"""
Decoupled Business Intelligence Analytics Module for Day 62.
Provides pure mathematical and statistical aggregations for dashboard visualization.
"""

import pandas as pd


def get_kpi_summary(df: pd.DataFrame) -> dict:
    """
    Computes top-level executive KPIs from transactional data.

    # What is used: sum(), nunique(), and safe margin division
    # Why it is used: Establishes the 4 fundamental health indicators of the business
    # How it works: Aggregates revenue, profit, orders, unique customers, and profit margin %
    """
    tot_rev = float(df["Revenue"].sum())
    tot_profit = float(df["Profit"].sum())
    tot_orders = int(df["Order_ID"].nunique()) if "Order_ID" in df.columns else len(df)
    tot_cust = int(df["Customer_ID"].nunique()) if "Customer_ID" in df.columns else 0
    margin_pct = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0.0

    return {
        "total_revenue": tot_rev,
        "total_profit": tot_profit,
        "total_orders": tot_orders,
        "total_customers": tot_cust,
        "profit_margin": margin_pct
    }


def get_monthly_revenue(df: pd.DataFrame) -> pd.Series:
    """Calculates total revenue per month sorted chronologically."""
    col = "Year_Month" if "Year_Month" in df.columns else "Month"
    return df.groupby(col)["Revenue"].sum().sort_index()


def get_regional_revenue(df: pd.DataFrame) -> pd.Series:
    """Calculates regional revenue sorted descending."""
    return df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)


def get_category_revenue(df: pd.DataFrame) -> pd.Series:
    """Calculates category revenue sorted descending."""
    return df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)


def get_top_products(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """Extracts top N products sorted ascending for horizontal bar rendering."""
    return df.groupby("Product")["Revenue"].sum().sort_values(ascending=True).tail(n)


def get_revenue_vs_profit(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Extracts paired revenue and profit vectors and computes Pearson correlation r."""
    clean_df = df[["Revenue", "Profit"]].dropna()
    r = float(clean_df["Revenue"].corr(clean_df["Profit"]))
    return clean_df["Revenue"], clean_df["Profit"], r


def get_quantity_distribution(df: pd.DataFrame) -> pd.Series:
    """Extracts non-null transaction order quantities."""
    return df["Quantity"].dropna()


def get_monthly_rolling_revenue(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Computes monthly revenue and a 3-month moving average."""
    col = "Year_Month" if "Year_Month" in df.columns else "Month"
    monthly = df.groupby(col)["Revenue"].sum().reset_index()
    monthly["Rolling_Avg"] = monthly["Revenue"].rolling(window=window, min_periods=1).mean()
    return monthly