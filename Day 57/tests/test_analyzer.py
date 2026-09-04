"""
Unit tests for business analytical engine app/analyzer.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 57 parent directory to sys.path.
import sys
from pathlib import Path

DAY57_DIR = Path(__file__).resolve().parent.parent
if str(DAY57_DIR) not in sys.path:
    sys.path.insert(0, str(DAY57_DIR))

# What is used: Import pandas and generate_analysis function.
# Why it is used: Asserts business metrics, regional/category rankings, top customers, and pivot tables.
# How it works: Executes generate_analysis on sample clean sales DataFrame.
import pandas as pd
from app.analyzer import generate_analysis


def test_analysis_overall_metrics(sample_clean_sales_df):
    """
    Test calculation of overall business metrics (Total Revenue, Orders, AOV).
    """
    # What is used: generate_analysis call.
    # Why it is used: Verifies overall total revenue, order count, unique customer count, and AOV.
    # How it works: Compares calculated overall metrics dictionary values against expected amounts.
    results, _ = generate_analysis(sample_clean_sales_df)
    overall = results["overall"]
    # Total Revenue: 114000 + 81000 + 20000 + 38250 = 253250.0
    assert overall["total_revenue"] == 253250.0
    assert overall["total_orders"] == 4
    assert overall["total_customers"] == 4
    assert overall["average_order_value"] == round(253250.0 / 4, 2)


def test_analysis_regional_revenue(sample_clean_sales_df):
    """
    Test regional revenue breakdown and top/bottom region identification.
    """
    # What is used: generate_analysis execution for regional summary.
    # Why it is used: Asserts top region is West (114000.0) and bottom is South (20000.0).
    # How it works: Checks top_region and bottom_region keys in results dictionary.
    results, _ = generate_analysis(sample_clean_sales_df)
    region_info = results["region"]
    assert region_info["top_region"] == "West"
    assert region_info["bottom_region"] == "South"


def test_analysis_category_revenue(sample_clean_sales_df):
    """
    Test category revenue breakdown and top category identification.
    """
    # What is used: generate_analysis execution for category summary.
    # Why it is used: Asserts Electronics is top performing category.
    # How it works: Checks top_category string value.
    results, _ = generate_analysis(sample_clean_sales_df)
    category_info = results["category"]
    assert category_info["top_category"] == "Electronics"


def test_analysis_product_revenue(sample_clean_sales_df):
    """
    Test product revenue ranking and total units sold calculation.
    """
    # What is used: generate_analysis execution for product metrics.
    # Why it is used: Asserts Laptop is best revenue product and total units sold is 11.
    # How it works: Verifies product summary dictionary values.
    results, _ = generate_analysis(sample_clean_sales_df)
    product_info = results["product"]
    assert product_info["best_product_revenue"] == "Laptop"
    assert product_info["total_units_sold"] == 11  # 2 + 3 + 5 + 1


def test_analysis_top_customer(sample_clean_sales_df):
    """
    Test top customer by revenue identification.
    """
    # What is used: generate_analysis execution for customer insights.
    # Why it is used: Asserts Rahul Sharma is top customer by total revenue.
    # How it works: Checks top_customer_revenue key in customer results.
    results, _ = generate_analysis(sample_clean_sales_df)
    customer_info = results["customer"]
    assert customer_info["top_customer_revenue"] == "Rahul Sharma"


def test_analysis_pivot_table(sample_clean_sales_df):
    """
    Test Region x Category pivot table generation.
    """
    # What is used: generate_analysis execution for 2D pivot table.
    # Why it is used: Verifies pivot table DataFrame shape, index, columns, and cell values.
    # How it works: Checks pivot_df.loc["West", "Electronics"] == 114000.0.
    _, pivot_df = generate_analysis(sample_clean_sales_df)
    assert isinstance(pivot_df, pd.DataFrame)
    assert "Electronics" in pivot_df.columns
    assert "West" in pivot_df.index
    assert pivot_df.loc["West", "Electronics"] == 114000.0


def test_analysis_monthly_time_series(sample_clean_sales_df):
    """
    Test monthly time-series aggregation.
    """
    # What is used: generate_analysis execution for monthly trends.
    # Why it is used: Asserts best month is 2026-01 (revenue 215000.0) vs 2026-02 (38250.0).
    # How it works: Checks best_month and worst_month keys.
    results, _ = generate_analysis(sample_clean_sales_df)
    monthly_info = results["monthly"]
    assert monthly_info["best_month"] == "2026-01"
    assert monthly_info["worst_month"] == "2026-02"


def test_analysis_top_and_bottom_orders(sample_clean_sales_df):
    """
    Test top 10 and bottom 10 order extraction.
    """
    # What is used: generate_analysis execution for nlargest and nsmallest order extraction.
    # Why it is used: Asserts order ranking lists and largest order revenue.
    # How it works: Checks orders dictionary keys.
    results, _ = generate_analysis(sample_clean_sales_df)
    orders_info = results["orders"]
    assert orders_info["largest_order_revenue"] == 114000.0
    assert len(orders_info["top_10"]) == 4
    assert len(orders_info["bottom_10"]) == 4
