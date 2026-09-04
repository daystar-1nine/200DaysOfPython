"""
Module: analyzer.py
Computes comprehensive business analytics, regional/category breakdowns, customer insights, and pivot tables.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core package for data aggregation, grouping, and matrix pivoting.
# How it works: Brings pandas and numpy namespaces into scope.
import numpy as np
import pandas as pd


def generate_analysis(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Perform full business sales analytics on transformed DataFrame.

    Args:
        df: Transformed sales DataFrame.

    Returns:
        tuple[dict, pd.DataFrame]: Analytics results dictionary and Region x Category pivot table.
    """
    results = {}

    # 1. Overall Metrics
    total_revenue = float(df["Revenue"].sum())
    total_orders = len(df)
    total_customers = int(df["Customer_ID"].nunique())
    aov = round(total_revenue / total_orders, 2) if total_orders > 0 else 0.0

    results["overall"] = {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "average_order_value": aov
    }

    # 2. Regional Analysis
    # What is used: df.groupby("Region")["Revenue"].sum().reset_index().
    # Why it is used: Aggregates revenue by geographic region.
    # How it works: Sums revenue per region and calculates percentage contribution.
    region_summary = df.groupby("Region")["Revenue"].sum().reset_index()
    region_summary["Percentage"] = (region_summary["Revenue"] / total_revenue * 100.0).round(2) if total_revenue > 0 else 0.0
    region_summary_sorted = region_summary.sort_values(by="Revenue", ascending=False)

    top_region = str(region_summary_sorted.iloc[0]["Region"]) if len(region_summary_sorted) > 0 else "N/A"
    bottom_region = str(region_summary_sorted.iloc[-1]["Region"]) if len(region_summary_sorted) > 0 else "N/A"

    results["region"] = {
        "summary": region_summary_sorted.to_dict(orient="records"),
        "top_region": top_region,
        "bottom_region": bottom_region
    }

    # 3. Category Analysis
    category_summary = df.groupby("Category")["Revenue"].sum().reset_index()
    category_summary["Percentage"] = (category_summary["Revenue"] / total_revenue * 100.0).round(2) if total_revenue > 0 else 0.0
    category_sorted = category_summary.sort_values(by="Revenue", ascending=False)

    top_category = str(category_sorted.iloc[0]["Category"]) if len(category_sorted) > 0 else "N/A"
    bottom_category = str(category_sorted.iloc[-1]["Category"]) if len(category_sorted) > 0 else "N/A"

    results["category"] = {
        "summary": category_sorted.to_dict(orient="records"),
        "top_category": top_category,
        "bottom_category": bottom_category
    }

    # 4. Product Analysis
    # What is used: Named aggregation for product revenue and quantity.
    # Why it is used: Calculates total revenue, units sold, and order count per product.
    # How it works: Groups by Product and aggregates sum/count.
    prod_summary = df.groupby("Product").agg(
        total_revenue=("Revenue", "sum"),
        total_units=("Quantity", "sum"),
        order_count=("Order_ID", "count")
    ).reset_index()

    prod_by_rev = prod_summary.sort_values(by="total_revenue", ascending=False)
    best_product_rev = str(prod_by_rev.iloc[0]["Product"]) if len(prod_by_rev) > 0 else "N/A"
    lowest_product_rev = str(prod_by_rev.iloc[-1]["Product"]) if len(prod_by_rev) > 0 else "N/A"

    prod_by_units = prod_summary.sort_values(by="total_units", ascending=False)
    best_product_units = str(prod_by_units.iloc[0]["Product"]) if len(prod_by_units) > 0 else "N/A"

    results["product"] = {
        "summary": prod_by_rev.to_dict(orient="records"),
        "best_product_revenue": best_product_rev,
        "lowest_product_revenue": lowest_product_rev,
        "best_product_units": best_product_units,
        "total_units_sold": int(df["Quantity"].sum())
    }

    # 5. Customer Analysis
    cust_summary = df.groupby("Customer_Name").agg(
        total_revenue=("Revenue", "sum"),
        order_count=("Order_ID", "count"),
        average_order=("Revenue", "mean")
    ).reset_index()

    cust_sorted = cust_summary.sort_values(by="total_revenue", ascending=False)
    top_customer_rev = str(cust_sorted.iloc[0]["Customer_Name"]) if len(cust_sorted) > 0 else "N/A"

    cust_by_orders = cust_summary.sort_values(by="order_count", ascending=False)
    top_customer_orders = str(cust_by_orders.iloc[0]["Customer_Name"]) if len(cust_by_orders) > 0 else "N/A"

    results["customer"] = {
        "summary": cust_sorted.head(10).to_dict(orient="records"),
        "top_customer_revenue": top_customer_rev,
        "top_customer_orders": top_customer_orders
    }

    # 6. Regional x Category Combination Analysis
    reg_cat = df.groupby(["Region", "Category"])["Revenue"].sum().reset_index()
    reg_cat_sorted = reg_cat.sort_values(by="Revenue", ascending=False)
    best_reg_cat = f"{reg_cat_sorted.iloc[0]['Region']} / {reg_cat_sorted.iloc[0]['Category']}" if len(reg_cat_sorted) > 0 else "N/A"

    results["region_category"] = {
        "summary": reg_cat_sorted.to_dict(orient="records"),
        "best_combination": best_reg_cat
    }

    # 7. Pivot Table (Region x Category)
    # What is used: pd.pivot_table().
    # Why it is used: Creates 2D matrix of Region (index) by Category (columns) with revenue sums.
    # How it works: Groups data into tabular grid with 0 fill values for missing cells.
    pivot_df = pd.pivot_table(
        df,
        values="Revenue",
        index="Region",
        columns="Category",
        aggfunc="sum",
        fill_value=0.0
    ).round(2)

    # 8. Discount Analysis
    avg_discount = float(df["Discount"].mean() * 100.0)
    disc_summary = df.groupby("Discount_Range", observed=False).agg(
        total_revenue=("Revenue", "sum"),
        order_count=("Order_ID", "count"),
        average_revenue=("Revenue", "mean")
    ).reset_index()

    results["discount"] = {
        "average_discount_pct": round(avg_discount, 2),
        "range_summary": disc_summary.to_dict(orient="records")
    }

    # 9. Top 10 & Bottom 10 Orders
    top_10_orders = df.nlargest(10, "Revenue")[["Order_ID", "Customer_Name", "Region", "Category", "Product", "Revenue"]].to_dict(orient="records")
    bottom_10_orders = df.nsmallest(10, "Revenue")[["Order_ID", "Customer_Name", "Region", "Category", "Product", "Revenue"]].to_dict(orient="records")

    results["orders"] = {
        "top_10": top_10_orders,
        "bottom_10": bottom_10_orders,
        "largest_order_revenue": float(df["Revenue"].max()) if len(df) > 0 else 0.0
    }

    # 10. Monthly Analysis (Time-Series)
    monthly_summary = df.groupby("Month")["Revenue"].agg(["sum", "count", "mean"]).reset_index()
    monthly_summary.rename(columns={"sum": "Revenue", "count": "Orders", "mean": "AOV"}, inplace=True)
    monthly_sorted = monthly_summary.sort_values(by="Revenue", ascending=False)

    best_month = str(monthly_sorted.iloc[0]["Month"]) if len(monthly_sorted) > 0 else "N/A"
    worst_month = str(monthly_sorted.iloc[-1]["Month"]) if len(monthly_sorted) > 0 else "N/A"

    results["monthly"] = {
        "summary": monthly_summary.to_dict(orient="records"),
        "best_month": best_month,
        "worst_month": worst_month
    }

    return results, pivot_df
