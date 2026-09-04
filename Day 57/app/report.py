"""
Module: report.py
Generates ASCII formatted executive business reports and exports analytical CSV datasets.
"""

# What is used: Import os module and pathlib Path class.
# Why it is used: Manages output directory paths and file writing operations safely.
# How it works: Ensures output directories exist before exporting files.
import os
from pathlib import Path

# What is used: Import pandas library.
# Why it is used: Handles CSV exporting via df.to_csv().
# How it works: Saves DataFrame objects to disk.
import pandas as pd


def generate_report(results: dict, cleaning_stats: dict) -> str:
    """
    Format analytical results and audit statistics into an executive ASCII text report.

    Args:
        results: Analytics results dictionary from analyzer.
        cleaning_stats: Cleaning audit statistics from cleaner.

    Returns:
        str: Formatted multiline ASCII text report string.
    """
    overall = results.get("overall", {})
    region_info = results.get("region", {})
    category_info = results.get("category", {})
    product_info = results.get("product", {})
    customer_info = results.get("customer", {})
    reg_cat_info = results.get("region_category", {})
    discount_info = results.get("discount", {})
    orders_info = results.get("orders", {})
    monthly_info = results.get("monthly", {})

    report_lines = [
        "================================================================================",
        "                      SALES ANALYTICS ENGINE - EXECUTIVE REPORT                  ",
        "================================================================================",
        "",
        "--- 1. DATA CLEANING AUDIT METRICS ---",
        f"  * Initial Raw Records   : {cleaning_stats.get('initial_rows', 0)}",
        f"  * Duplicates Removed    : {cleaning_stats.get('duplicates_removed', 0)}",
        f"  * Missing Values Imputed: {cleaning_stats.get('nulls_filled', 0)}",
        f"  * Invalid Records Dropped: {cleaning_stats.get('invalid_rows_dropped', 0)}",
        f"  * Clean Analytical Rows : {cleaning_stats.get('final_rows', 0)}",
        "",
        "--- 2. EXECUTIVE BUSINESS SUMMARY ---",
        f"  * Total Grand Revenue   : ${overall.get('total_revenue', 0.0):,.2f}",
        f"  * Total Orders Processed : {overall.get('total_orders', 0)}",
        f"  * Unique Active Customers: {overall.get('total_customers', 0)}",
        f"  * Average Order Value   : ${overall.get('average_order_value', 0.0):,.2f}",
        f"  * Largest Order Revenue : ${orders_info.get('largest_order_revenue', 0.0):,.2f}",
        "",
        "--- 3. REGIONAL PERFORMANCE ANALYSIS ---",
        f"  * Top Performing Region : {region_info.get('top_region', 'N/A')}",
        f"  * Lowest Revenue Region : {region_info.get('bottom_region', 'N/A')}",
        "  * Regional Breakdown:",
        f"    {'Region':<12} {'Revenue ($)':<16} {'Contribution (%)':<18}",
        "    " + "-" * 48
    ]

    for reg in region_info.get("summary", []):
        pct_str = f"{reg.get('Percentage', 0.0):.2f}%"
        report_lines.append(
            f"    {reg.get('Region', 'N/A'):<12} ${reg.get('Revenue', 0.0):<15,.2f} {pct_str:<18}"
        )

    report_lines.extend([
        "",
        "--- 4. CATEGORY PERFORMANCE ANALYSIS ---",
        f"  * Top Performing Category: {category_info.get('top_category', 'N/A')}",
        f"  * Lowest Revenue Category: {category_info.get('bottom_category', 'N/A')}",
        f"  * Best Region/Category   : {reg_cat_info.get('best_combination', 'N/A')}",
        "  * Category Breakdown:",
        f"    {'Category':<16} {'Revenue ($)':<16} {'Contribution (%)':<18}",
        "    " + "-" * 52
    ])

    for cat in category_info.get("summary", []):
        pct_str = f"{cat.get('Percentage', 0.0):.2f}%"
        report_lines.append(
            f"    {cat.get('Category', 'N/A'):<16} ${cat.get('Revenue', 0.0):<15,.2f} {pct_str:<18}"
        )

    report_lines.extend([
        "",
        "--- 5. PRODUCT & INVENTORY INSIGHTS ---",
        f"  * Top Product by Revenue: {product_info.get('best_product_revenue', 'N/A')}",
        f"  * Lowest Product Revenue: {product_info.get('lowest_product_revenue', 'N/A')}",
        f"  * Top Product by Volume : {product_info.get('best_product_units', 'N/A')}",
        f"  * Total Units Sold      : {product_info.get('total_units_sold', 0):,}",
        "",
        "--- 6. CUSTOMER SEGMENTATION (TOP 5 HIGHEST VALUE) ---",
        f"  * Highest Revenue Customer: {customer_info.get('top_customer_revenue', 'N/A')}",
        f"  * Most Frequent Orders    : {customer_info.get('top_customer_orders', 'N/A')}",
        f"    {'Customer Name':<20} {'Total Rev ($)':<15} {'Orders':<8} {'Avg Order ($)':<14}",
        "    " + "-" * 60
    ])

    for cust in customer_info.get("summary", [])[:5]:
        report_lines.append(
            f"    {cust.get('Customer_Name', 'N/A'):<20} ${cust.get('total_revenue', 0.0):<14,.2f} "
            f"{cust.get('order_count', 0):<8} ${cust.get('average_order', 0.0):<14,.2f}"
        )

    report_lines.extend([
        "",
        "--- 7. DISCOUNT IMPACT ANALYSIS ---",
        f"  * Class Average Discount: {discount_info.get('average_discount_pct', 0.0)}%",
        "  * Discount Tier Breakdown:",
        f"    {'Discount Tier':<20} {'Revenue ($)':<16} {'Orders':<8} {'Avg Rev/Order ($)':<18}",
        "    " + "-" * 64
    ])

    for disc in discount_info.get("range_summary", []):
        report_lines.append(
            f"    {str(disc.get('Discount_Range', 'N/A')):<20} ${disc.get('total_revenue', 0.0):<15,.2f} "
            f"{disc.get('order_count', 0):<8} ${disc.get('average_revenue', 0.0):<18,.2f}"
        )

    report_lines.extend([
        "",
        "--- 8. MONTHLY TIME-SERIES TRENDS ---",
        f"  * Peak Sales Month      : {monthly_info.get('best_month', 'N/A')}",
        f"  * Lowest Sales Month    : {monthly_info.get('worst_month', 'N/A')}",
        f"    {'Month':<10} {'Revenue ($)':<16} {'Orders':<8} {'Avg Order Value ($)':<18}",
        "    " + "-" * 56
    ])

    for m in monthly_info.get("summary", []):
        report_lines.append(
            f"    {str(m.get('Month', 'N/A')):<10} ${m.get('Revenue', 0.0):<15,.2f} "
            f"{m.get('Orders', 0):<8} ${m.get('AOV', 0.0):<18,.2f}"
        )

    report_lines.append("================================================================================")
    return "\n".join(report_lines)


def export_reports_and_summaries(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    report_str: str,
    cleaned_csv_path: str | Path,
    report_path: str | Path,
    pivot_csv_path: str | Path
) -> None:
    """
    Export cleaned sales CSV, executive ASCII report text file, and regional pivot summary CSV.

    Args:
        df: Cleaned and transformed sales DataFrame.
        pivot_df: Region x Category pivot table DataFrame.
        report_str: Formatted executive ASCII report.
        cleaned_csv_path: Path for cleaned sales dataset export.
        report_path: Path for text report export.
        pivot_csv_path: Path for regional pivot table CSV export.
    """
    # What is used: Path.mkdir(parents=True, exist_ok=True).
    # Why it is used: Guarantees destination parent folders exist before writing files.
    # How it works: Recursively creates parent directory tree if missing.
    c_path = Path(cleaned_csv_path)
    r_path = Path(report_path)
    p_path = Path(pivot_csv_path)

    c_path.parent.mkdir(parents=True, exist_ok=True)
    r_path.parent.mkdir(parents=True, exist_ok=True)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    # What is used: df.to_csv(index=False).
    # Why it is used: Writes cleaned dataset and pivot table to CSV files.
    # How it works: Serializes DataFrames to disk.
    df.to_csv(c_path, index=False)
    pivot_df.to_csv(p_path)

    # What is used: File open() with encoding='utf-8'.
    # Why it is used: Writes executive report string to text file safely.
    # How it works: Saves ASCII report to disk.
    with open(r_path, "w", encoding="utf-8") as f:
        f.write(report_str)
