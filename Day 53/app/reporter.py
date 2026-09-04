"""
===============================================================================
DAY 53 — REPORT GENERATOR & CLEAN CSV EXPORTER MODULE
===============================================================================
This module formats analytical summaries into ASCII text reports and exports
processed clean sales data to CSV format.
===============================================================================
"""

import csv
from pathlib import Path
from typing import List
from datetime import date
from app.models import Sale
from app.analyzer import (
    total_revenue,
    average_order_value,
    highest_value_order,
    best_selling_product,
    category_revenue,
    top_products_by_revenue,
)


def export_cleaned_csv(sales: List[Sale], output_path: Path) -> None:
    """Export cleaned, validated, and deduplicated Sale records to CSV file."""
    # What is used: Path.parent.mkdir and csv.DictWriter.
    # Why it is used: Persists processed data layer for downstream analytics tools.
    # How it works: Writes header row and iterates sale.to_dict() dictionaries.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["order_id", "customer", "product", "category", "price", "quantity", "date", "total"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sale in sales:
            writer.writerow(sale.to_dict())


def generate_sales_report(
    sales: List[Sale],
    raw_count: int,
    invalid_count: int,
    duplicate_count: int,
    report_path: Path,
) -> str:
    """Generate and write a formatted ASCII sales analysis report."""
    # What is used: Formatted multi-line string interpolation and Path file writing.
    # Why it is used: Creates human-readable business executive summaries.
    # How it works: Computes analytical metrics, formats ASCII sections, and saves file.
    tot_rev = total_revenue(sales)
    aov = average_order_value(sales)
    highest_order = highest_value_order(sales)
    top_prod, top_qty = best_selling_product(sales)
    cat_rev = category_revenue(sales)
    top_5 = top_products_by_revenue(sales, top_n=5)

    today_str = date.today().strftime("%Y-%m-%d")

    lines = [
        "========================================",
        "          SALES ANALYSIS REPORT         ",
        "========================================",
        f"Generated: {today_str}",
        "",
        "DATASET PIPELINE AUDIT",
        "----------------------------------------",
        f"Raw Records:          {raw_count}",
        f"Valid Records:        {len(sales)}",
        f"Invalid Records:      {invalid_count}",
        f"Duplicates Removed:   {duplicate_count}",
        "",
        "SALES FINANCIAL SUMMARY",
        "----------------------------------------",
        f"Total Orders:         {len(sales)}",
        f"Total Revenue:        ${tot_rev:,.2f}",
        f"Average Order Value:  ${aov:,.2f}",
        "",
        "HIGHEST VALUE TRANSACTION",
        "----------------------------------------",
    ]

    if highest_order:
        lines.extend([
            f"Order ID:  #{highest_order.order_id}",
            f"Customer:  {highest_order.customer}",
            f"Product:   {highest_order.product}",
            f"Amount:    ${highest_order.total:,.2f}",
        ])
    else:
        lines.append("No orders processed.")

    lines.extend([
        "",
        "TOP SELLING PRODUCT (BY QUANTITY)",
        "----------------------------------------",
        f"{top_prod} ({top_qty} units sold)",
        "",
        "CATEGORY PERFORMANCE",
        "----------------------------------------",
    ])

    for cat, rev in cat_rev.items():
        lines.append(f"{cat:<15}: ${rev:,.2f}")

    lines.extend([
        "",
        "TOP 5 PRODUCTS (BY REVENUE)",
        "----------------------------------------",
    ])

    for idx, (p_name, p_rev) in enumerate(top_5, start=1):
        lines.append(f"{idx}. {p_name:<15} -> ${p_rev:,.2f}")

    lines.append("========================================\n")

    report_content = "\n".join(lines)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")

    return report_content
