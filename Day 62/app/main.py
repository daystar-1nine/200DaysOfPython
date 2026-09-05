"""
Main Application Entry Point for Day 62.
Coordinates data ingestion, standalone chart exports, and master dashboard generation.
"""

import sys
from pathlib import Path

# Configure Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import (
    ECOMMERCE_CSV_PATH,
    CHARTS_DIR,
    DASHBOARD_PATH,
    FIGSIZE_STANDARD,
    FIGSIZE_WIDE,
    FIGSIZE_HORIZONTAL
)
from app.loader import load_ecommerce_data
from app import analyzer
from app import charts
from app.dashboard import create_dashboard


def run_pipeline() -> None:
    """Executes the complete Day 62 dashboard and visualization generation pipeline."""
    print("=" * 70)
    print("  🚀 DAY 62: E-COMMERCE PERFORMANCE DASHBOARD ENGINE")
    print("=" * 70)

    # 1. Ingestion
    print("\n[Phase 1/3] Loading and validating e-commerce dataset...")
    df = load_ecommerce_data(ECOMMERCE_CSV_PATH)
    print(f"  ✓ Successfully loaded {len(df):,} transaction records.")

    # 2. Individual Charts Export
    print("\n[Phase 2/3] Generating individual high-resolution charts...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Monthly Revenue
    m_rev = analyzer.get_monthly_revenue(df)
    fig1, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
    charts.plot_monthly_revenue_trend(ax1, m_rev, avg_line=True, annotate_max=True)
    c1 = charts.save_individual_chart(fig1, CHARTS_DIR / "monthly_revenue.png")
    print(f"  ✓ [1/6] Saved: {c1.name}")

    # Regional Revenue
    reg_rev = analyzer.get_regional_revenue(df)
    fig2, ax2 = plt.subplots(figsize=FIGSIZE_STANDARD)
    charts.plot_regional_revenue(ax2, reg_rev)
    c2 = charts.save_individual_chart(fig2, CHARTS_DIR / "regional_revenue.png")
    print(f"  ✓ [2/6] Saved: {c2.name}")

    # Category Revenue
    cat_rev = analyzer.get_category_revenue(df)
    fig3, ax3 = plt.subplots(figsize=FIGSIZE_STANDARD)
    charts.plot_category_revenue(ax3, cat_rev)
    c3 = charts.save_individual_chart(fig3, CHARTS_DIR / "category_revenue.png")
    print(f"  ✓ [3/6] Saved: {c3.name}")

    # Top Products
    top_p = analyzer.get_top_products(df, n=10)
    fig4, ax4 = plt.subplots(figsize=FIGSIZE_HORIZONTAL)
    charts.plot_top_products(ax4, top_p)
    c4 = charts.save_individual_chart(fig4, CHARTS_DIR / "top_products.png")
    print(f"  ✓ [4/6] Saved: {c4.name}")

    # Revenue vs Profit Scatter
    rev_v, prof_v, r = analyzer.get_revenue_vs_profit(df)
    fig5, ax5 = plt.subplots(figsize=FIGSIZE_STANDARD)
    charts.plot_revenue_vs_profit(ax5, rev_v, prof_v, r)
    c5 = charts.save_individual_chart(fig5, CHARTS_DIR / "revenue_profit.png")
    print(f"  ✓ [5/6] Saved: {c5.name}")

    # Quantity Distribution
    qty_dist = analyzer.get_quantity_distribution(df)
    fig6, ax6 = plt.subplots(figsize=FIGSIZE_STANDARD)
    charts.plot_quantity_distribution(ax6, qty_dist)
    c6 = charts.save_individual_chart(fig6, CHARTS_DIR / "quantity_distribution.png")
    print(f"  ✓ [6/6] Saved: {c6.name}")

    # 3. Master Dashboard
    print("\n[Phase 3/3] Compiling Master Executive Analytics Dashboard...")
    dash_path = create_dashboard(df, DASHBOARD_PATH)
    print(f"  ✓ Master Dashboard Generated: {dash_path.name}")

    print("\n" + "=" * 70)
    print("  🎉 PIPELINE COMPLETE: All 6 charts and master dashboard compiled!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()