"""
Module: group_analysis.py
Performs grouped aggregations by Region, Category, Product, and Customer using agg(), transform(), and rank().
"""

# What is used: Import pandas library.
# Why it is used: Core package for grouped aggregations, intra-group transform, and ranking.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def analyze_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total revenue, average revenue, total profit, order count, and rank by Region.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Regional performance summary table.
    """
    res = (
        df.groupby("Region")
        .agg(
            total_revenue=("Revenue", "sum"),
            average_revenue=("Revenue", "mean"),
            total_profit=("Profit", "sum"),
            order_count=("Order_ID", "count")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["average_revenue"] = res["average_revenue"].round(2)
    res["total_profit"] = res["total_profit"].round(2)

    # What is used: Series.rank(ascending=False).
    # Why it is used: Assigns rank 1 to region with highest total revenue.
    # How it works: Ranks total_revenue in descending order.
    res["Revenue_Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)
    return res.sort_values(by="Revenue_Rank").reset_index(drop=True)


def analyze_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total revenue, profit, quantity sold, average discount, order count by Category.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Category performance summary table.
    """
    res = (
        df.groupby("Category")
        .agg(
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_discount=("Discount", "mean"),
            order_count=("Order_ID", "count")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["avg_discount"] = (res["avg_discount"] * 100.0).round(2)
    res["Revenue_Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    return res.sort_values(by="Revenue_Rank").reset_index(drop=True)


def analyze_product_performance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute product metrics, rankings, and identify Top 10 by Revenue, Quantity, and Profit.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (top_revenue, top_quantity, top_profit).
    """
    res = (
        df.groupby(["Category", "Product"])
        .agg(
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_unit_price=("Unit_Price", "mean")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["avg_unit_price"] = res["avg_unit_price"].round(2)

    # What is used: Category-level intra-group transform and ranking.
    # Why it is used: Ranks products within their specific category partitions.
    # How it works: Applies rank(ascending=False) per Category group.
    res["Category_Rank"] = (
        res.groupby("Category")["total_revenue"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    res["Overall_Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    top_revenue = res.sort_values(by="total_revenue", ascending=False).head(10).reset_index(drop=True)
    top_quantity = res.sort_values(by="total_quantity", ascending=False).head(10).reset_index(drop=True)
    top_profit = res.sort_values(by="total_profit", ascending=False).head(10).reset_index(drop=True)

    return top_revenue, top_quantity, top_profit


def analyze_customer_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute customer spend metrics, total profit, order counts, average order value (AOV), and customer ranks.

    Args:
        df: Enriched sales DataFrame.

    Returns:
        pd.DataFrame: Customer summary table sorted by total revenue descending.
    """
    res = (
        df.groupby(["Customer_ID", "Customer_Name"])
        .agg(
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order_ID", "count"),
            aov=("Revenue", "mean")
        )
        .reset_index()
    )

    res["total_revenue"] = res["total_revenue"].round(2)
    res["total_profit"] = res["total_profit"].round(2)
    res["aov"] = res["aov"].round(2)

    res["Customer_Rank"] = res["total_revenue"].rank(ascending=False, method="dense").astype(int)

    # What is used: transform("mean") for customer spend comparison.
    # Why it is used: Broadcasts overall average customer revenue across rows for relative comparison.
    # How it works: Compares total_revenue against average customer spend.
    avg_spend = res["total_revenue"].mean()
    res["Above_Average_Customer"] = res["total_revenue"] > avg_spend

    return res.sort_values(by="Customer_Rank").reset_index(drop=True)
