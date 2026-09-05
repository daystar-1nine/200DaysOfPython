"""
Decoupled Statistical and Aggregation Analysis Module.
Contains pure data transformation and analytical functions for downstream visualization.
"""

import pandas as pd


def get_monthly_revenue(df: pd.DataFrame) -> pd.Series:
    """
    Calculates total revenue aggregated by month.

    # What is used: groupby on Year_Month with sum aggregation
    # Why it is used: Provides chronological sequence of top-line revenue performance
    # How it works: Groups records chronologically and computes total revenue per period
    """
    return df.groupby("Year_Month")["Revenue"].sum().sort_index()


def get_revenue_by_region(df: pd.DataFrame) -> pd.Series:
    """
    Calculates total revenue aggregated by geographic sales region sorted descending.

    # What is used: groupby on Region with sort_values(ascending=False)
    # Why it is used: Enables immediate comparison of geographic market share
    # How it works: Sums revenue per region and sorts from highest to lowest
    """
    return df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)


def get_revenue_by_category(df: pd.DataFrame) -> pd.Series:
    """
    Calculates total revenue aggregated by product category sorted descending.

    # What is used: groupby on Category with sort_values(ascending=False)
    # Why it is used: Uncovers the strongest and weakest business product lines
    # How it works: Sums revenue per category and sorts from highest to lowest
    """
    return df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)


def get_top_n_products(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    Calculates total revenue per product and returns the top N highest performers.

    # What is used: groupby on Product with nlargest(n)
    # Why it is used: Focuses product inventory and marketing strategy on leading SKUs
    # How it works: Groups by product, sums revenue, and extracts top n items
    """
    return df.groupby("Product")["Revenue"].sum().sort_values(ascending=True).tail(n)


def get_top_n_customers(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    Calculates total revenue per customer and returns the top N VIP accounts.

    # What is used: groupby on Customer_Name with nlargest(n)
    # Why it is used: Identifies core whale accounts driving revenue concentration
    # How it works: Sums spending per customer and extracts top n records sorted ascending for horizontal bar
    """
    return df.groupby("Customer_Name")["Revenue"].sum().sort_values(ascending=True).tail(n)


def get_quantity_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Extracts transaction order quantities for distribution analysis.

    # What is used: Series extraction with dropna
    # Why it is used: Isolates discrete order volume data for histogram binning
    # How it works: Returns quantity series after dropping nulls
    """
    return df["Quantity"].dropna()


def get_revenue_vs_profit(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """
    Extracts Revenue and Profit pairs along with their Pearson correlation coefficient.

    # What is used: Series pairing and corr() calculation
    # Why it is used: Examines relationship between top-line volume and bottom-line margin
    # How it works: Extracts vectors and computes Pearson correlation r
    """
    clean_sub = df[["Revenue", "Profit"]].dropna()
    r = clean_sub["Revenue"].corr(clean_sub["Profit"])
    return clean_sub["Revenue"], clean_sub["Profit"], r


def get_category_revenue_share(df: pd.DataFrame) -> pd.Series:
    """
    Computes percentage share of total revenue per category.

    # What is used: Proportional division over sum
    # Why it is used: Normalizes category contributions to sum to 100%
    # How it works: Divides category sum by total sales revenue
    """
    cat_rev = df.groupby("Category")["Revenue"].sum()
    return (cat_rev / cat_rev.sum()) * 100


def get_monthly_rolling_revenue(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Computes monthly revenue along with a moving rolling average over specified window.

    # What is used: rolling(window).mean()
    # Why it is used: Filters out high-frequency monthly fluctuations to highlight momentum
    # How it works: Groups by month and applies rolling mean over sliding window
    """
    monthly = df.groupby("Year_Month")["Revenue"].sum().reset_index()
    monthly["Rolling_Avg"] = monthly["Revenue"].rolling(window=window, min_periods=1).mean()
    return monthly
