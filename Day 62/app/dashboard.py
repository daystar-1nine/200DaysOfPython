"""
Executive Master Dashboard Assembly Module for Day 62.
Uses matplotlib.gridspec.GridSpec to orchestrate a 4-tier asymmetric visual hierarchy.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless execution
# How it works: Sets "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from app.config import (
    DPI,
    FIGSIZE_DASHBOARD,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_SECONDARY,
    COLOR_PURPLE
)
from app.formatters import format_compact_inr, format_large_number, format_percentage
from app import analyzer
from app import charts


def create_dashboard(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Constructs the end-to-end static executive analytics dashboard using GridSpec.

    # What is used: matplotlib.gridspec.GridSpec with 4 tiered rows
    # Why it is used: Produces an executive-level single-page report adhering to visual hierarchy
    # How it works:
        - Row 0 (Cols 0-3): 4 KPI Metric Cards (Revenue, Profit, Orders, Margin)
        - Row 1 (Cols 0-3): Wide Primary Monthly Trend with Average Line and Peak Annotation
        - Row 2 (Cols 0-1 & 2-3): Regional and Category Revenue Comparisons
        - Row 3 (Cols 0-1 & 2-3): Top 10 SKUs and Bivariate Revenue vs Profit Association
    """
    # 1. Compute Analytics
    kpi = analyzer.get_kpi_summary(df)
    monthly_rev = analyzer.get_monthly_revenue(df)
    reg_rev = analyzer.get_regional_revenue(df)
    cat_rev = analyzer.get_category_revenue(df)
    top_prods = analyzer.get_top_products(df, n=10)
    rev_vec, prof_vec, corr_r = analyzer.get_revenue_vs_profit(df)

    # 2. Setup Figure & Asymmetric GridSpec
    fig = plt.figure(figsize=FIGSIZE_DASHBOARD)
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.9, 2.2, 1.9, 1.9], hspace=0.38, wspace=0.28)

    # 3. Row 0: 4 Executive KPI Cards
    ax_kpi1 = fig.add_subplot(gs[0, 0])
    charts.plot_kpi_card(ax_kpi1, "Total Revenue", format_compact_inr(kpi["total_revenue"]), "Gross Volume", COLOR_PRIMARY)

    ax_kpi2 = fig.add_subplot(gs[0, 1])
    charts.plot_kpi_card(ax_kpi2, "Net Profit", format_compact_inr(kpi["total_profit"]), "Bottom-Line Gain", COLOR_SUCCESS)

    ax_kpi3 = fig.add_subplot(gs[0, 2])
    charts.plot_kpi_card(ax_kpi3, "Total Orders", format_large_number(kpi["total_orders"]), "Completed Checkouts", COLOR_SECONDARY)

    ax_kpi4 = fig.add_subplot(gs[0, 3])
    charts.plot_kpi_card(ax_kpi4, "Profit Margin", format_percentage(kpi["profit_margin"]), "Corporate Margin", COLOR_PURPLE)

    # 4. Row 1: Primary Macro Trend (Spanning all 4 columns)
    ax_trend = fig.add_subplot(gs[1, :])
    charts.plot_monthly_revenue_trend(ax_trend, monthly_rev, avg_line=True, annotate_max=True)

    # 5. Row 2: Regional Performance & Category Breakdown
    ax_reg = fig.add_subplot(gs[2, :2])
    charts.plot_regional_revenue(ax_reg, reg_rev)

    ax_cat = fig.add_subplot(gs[2, 2:])
    charts.plot_category_revenue(ax_cat, cat_rev)

    # 6. Row 3: Top 10 SKUs & Order-Level Profit Association
    ax_prod = fig.add_subplot(gs[3, :2])
    charts.plot_top_products(ax_prod, top_prods)

    ax_scat = fig.add_subplot(gs[3, 2:])
    charts.plot_revenue_vs_profit(ax_scat, rev_vec, prof_vec, corr_r)

    # Master Title
    fig.suptitle("E-COMMERCE COMMERCIAL PERFORMANCE & OPERATIONAL DASHBOARD", fontsize=16, fontweight="bold", y=0.985)

    # Save and Reclaim Memory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return output_path