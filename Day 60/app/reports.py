"""
Module: reports.py
Formats Executive ASCII summaries, Data Quality reports, and exports all CSV analysis artifacts.
"""

# What is used: Import pathlib Path and pandas library.
# Why it is used: Manages directory structures and handles text and CSV file persistence.
# How it works: Ensures output directories exist and writes ASCII reports and analytical CSV summaries.
from pathlib import Path
import pandas as pd


def generate_executive_summary(
    overview_kpis: dict,
    regional_df: pd.DataFrame,
    category_df: pd.DataFrame,
    top_revenue_prod: pd.DataFrame,
    customer_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    stats_dict: dict,
    insights: list[str]
) -> str:
    """
    Format executive summary text report.

    Args:
        overview_kpis: Corporate KPIs.
        regional_df: Regional summary.
        category_df: Category summary.
        top_revenue_prod: Top products by revenue.
        customer_df: Top customers.
        monthly_df: Monthly trends.
        stats_dict: Descriptive statistics.
        insights: List of dynamic insights.

    Returns:
        str: Formatted ASCII executive report.
    """
    lines = [
        "================================================================================",
        "          EXECUTIVE BUSINESS INTELLIGENCE ANALYTICS REPORT — DAY 60             ",
        "================================================================================",
        "",
        "--- 1. MACRO CORPORATE PERFORMANCE KPIS ---",
        f"  * Total Orders Ingested  : {overview_kpis['total_orders']}",
        f"  * Total Gross Revenue    : ₹{overview_kpis['total_revenue']:,.2f}",
        f"  * Total Cost of Goods    : ₹{overview_kpis['total_cost']:,.2f}",
        f"  * Total Gross Profit     : ₹{overview_kpis['total_profit']:,.2f}",
        f"  * Overall Profit Margin  : {overview_kpis['overall_margin_%']:.2f}%",
        f"  * Average Order Value    : ₹{overview_kpis['average_order_value']:,.2f}",
        "",
        "--- 2. REGIONAL REVENUE & MARGIN BREAKDOWN ---",
        f"    {'Region':<8} {'Revenue':<16} {'Profit':<16} {'Share (%)':<12} {'Orders':<8}",
        "    " + "-" * 62
    ]

    for _, r in regional_df.iterrows():
        lines.append(
            f"    {r['Region']:<8} ₹{r['total_revenue']:<15,.2f} ₹{r['total_profit']:<15,.2f} {r['Revenue_Share_%']:<11.2f}% {r['order_count']:<8}"
        )

    lines.extend([
        "",
        "--- 3. CATEGORY PERFORMANCE BREAKDOWN ---",
        f"    {'Category':<16} {'Revenue':<16} {'Profit':<16} {'Margin (%)':<12} {'Units':<8}",
        "    " + "-" * 70
    ])

    for _, c in category_df.iterrows():
        lines.append(
            f"    {c['Category']:<16} ₹{c['total_revenue']:<15,.2f} ₹{c['total_profit']:<15,.2f} {c['profit_margin_%']:<11.2f}% {c['total_quantity']:<8}"
        )

    lines.extend([
        "",
        "--- 4. TOP 5 PRODUCTS BY REVENUE ---"
    ])
    for idx, p in top_revenue_prod.head(5).iterrows():
        lines.append(
            f"  {idx+1}. [{p['Product']:<25}] Category: {p['Category']:<14} Revenue: ₹{p['total_revenue']:,.2f} | Units: {p['total_quantity']}"
        )

    lines.extend([
        "",
        "--- 5. TOP 5 CUSTOMERS BY LIFETIME VALUE ---"
    ])
    for idx, cust in customer_df.head(5).iterrows():
        lines.append(
            f"  {idx+1}. [{cust['Customer_Name']:<18}] ID: {cust['Customer_ID']} | Revenue: ₹{cust['total_revenue']:,.2f} | Orders: {cust['order_count']} | AOV: ₹{cust['aov']:,.2f}"
        )

    lines.extend([
        "",
        "--- 6. MONTHLY PERFORMANCE & 3-MONTH ROLLING AVERAGES ---"
    ])
    for _, m in monthly_df.iterrows():
        growth_str = f"{m['MoM_Growth_%']:+.2f}%" if pd.notna(m.get("MoM_Growth_%")) else "N/A"
        roll_str = f"₹{m['Rolling_3M_Avg']:,.2f}" if pd.notna(m.get("Rolling_3M_Avg")) else "N/A"
        lines.append(
            f"  * Month [{m['Year_Month']}]: Revenue = ₹{m['total_revenue']:<12,.2f} | MoM Growth = {growth_str:<9} | 3M Rolling Avg = {roll_str}"
        )

    lines.extend([
        "",
        "================================================================================",
        "                     AUTOMATED BUSINESS INSIGHTS AUDIT                          ",
        "================================================================================"
    ])
    for ins in insights:
        lines.append(f"  {ins}")

    lines.append("================================================================================")
    return "\n".join(lines)


def generate_data_quality_report(clean_audit: dict, val_results: dict, outliers_dict: dict) -> str:
    """
    Format data quality and validation audit report.

    Args:
        clean_audit: Cleaner audit statistics.
        val_results: Validator audit results.
        outliers_dict: Outlier metrics dictionary.

    Returns:
        str: Formatted ASCII data quality audit report.
    """
    lines = [
        "================================================================================",
        "                   DATA QUALITY & DOMAIN VALIDATION REPORT                      ",
        "================================================================================",
        "",
        "--- 1. INGESTION & CLEANING STATISTICS ---",
        f"  * Raw Records Ingested  : {clean_audit.get('raw_rows', 0)}",
        f"  * Duplicates Eliminated : {clean_audit.get('duplicates_removed', 0)}",
        f"  * Missing Slots Imputed : {clean_audit.get('nulls_filled', 0)}",
        f"  * Final Cleaned Records : {clean_audit.get('final_rows', 0)}",
        "",
        "--- 2. BUSINESS RULE VALIDATION AUDIT ---",
        f"  * Overall Quality Status: {'PASSED' if val_results.get('is_valid', False) else 'FAILED'}"
    ]

    for r_name, r_info in val_results.get("rules", {}).items():
        status = "PASS" if r_info["passed"] else "FAIL"
        lines.append(f"  * Rule [{r_name:<24}]: {status} (Violations: {r_info['violations']})")

    lines.extend([
        "",
        "--- 3. STATISTICAL OUTLIER DETECTION (IQR) ---"
    ])
    for col, out_info in outliers_dict.items():
        lines.append(
            f"  * Feature [{col:<12}]: Q1 = {out_info['q1']:<9} Q3 = {out_info['q3']:<9} IQR = {out_info['iqr']:<9} "
            f"Bounds = [{out_info['lower_bound']} to {out_info['upper_bound']}] | Outliers = {out_info['outlier_count']}"
        )

    lines.append("================================================================================")
    return "\n".join(lines)


def export_all_artifacts(
    cleaned_df: pd.DataFrame,
    exec_summary: str,
    data_quality_rpt: str,
    regional_df: pd.DataFrame,
    category_df: pd.DataFrame,
    product_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    processed_path: Path,
    output_dir: Path
) -> None:
    """
    Export all processed datasets, text reports, and CSV summaries to disk.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(processed_path, index=False)

    with open(output_dir / "executive_summary.txt", "w", encoding="utf-8") as f:
        f.write(exec_summary)

    with open(output_dir / "data_quality_report.txt", "w", encoding="utf-8") as f:
        f.write(data_quality_rpt)

    regional_df.to_csv(output_dir / "regional_analysis.csv", index=False)
    category_df.to_csv(output_dir / "category_analysis.csv", index=False)
    product_df.to_csv(output_dir / "product_analysis.csv", index=False)
    customer_df.to_csv(output_dir / "customer_analysis.csv", index=False)
    monthly_df.to_csv(output_dir / "monthly_analysis.csv", index=False)
