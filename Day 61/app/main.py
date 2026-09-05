"""
Main Application Pipeline Orchestrator for Day 61.
Coordinates data ingestion, decoupled analytical computation, chart generation,
and executive report compilation.
"""

import sys
from pathlib import Path

# Configure Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import SALES_CSV_PATH, CHARTS_DIR, REPORT_PATH
from app.loader import load_sales_data
from app import analyzer
from app import charts
from app.report import generate_visualization_report


def run_pipeline() -> None:
    """Executes the complete end-to-end sales visualization pipeline."""
    print("=" * 70)
    print("  🚀 DAY 61: ENTERPRISE SALES VISUALIZATION REPORT ENGINE")
    print("=" * 70)

    # 1. Load data
    print("\n[Phase 1/4] Ingesting and validating sales dataset...")
    df = load_sales_data(SALES_CSV_PATH)
    print(f"  ✓ Loaded {len(df):,} validated transactions across {df['Category'].nunique()} categories.")

    # 2. Compute analytical metrics
    print("\n[Phase 2/4] Performing analytical aggregations...")
    monthly_rev = analyzer.get_monthly_revenue(df)
    reg_rev = analyzer.get_revenue_by_region(df)
    cat_rev = analyzer.get_revenue_by_category(df)
    top_prods = analyzer.get_top_n_products(df, n=10)
    top_custs = analyzer.get_top_n_customers(df, n=10)
    qty_dist = analyzer.get_quantity_distribution(df)
    rev_series, prof_series, corr_r = analyzer.get_revenue_vs_profit(df)
    cat_share = analyzer.get_category_revenue_share(df)
    monthly_rolling = analyzer.get_monthly_rolling_revenue(df, window=3)
    print("  ✓ Computed 9 analytical metric models successfully.")

    # 3. Generate charts
    print("\n[Phase 3/4] Generating publication-quality charts (300 DPI)...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    c1 = charts.plot_monthly_revenue(monthly_rev, CHARTS_DIR / "monthly_revenue.png")
    print(f"  ✓ [1/9] Line Chart Saved: {c1.name}")

    c2 = charts.plot_revenue_by_region(reg_rev, CHARTS_DIR / "regional_revenue.png")
    print(f"  ✓ [2/9] Regional Bar Chart Saved: {c2.name}")

    c3 = charts.plot_revenue_by_category(cat_rev, CHARTS_DIR / "category_revenue.png")
    print(f"  ✓ [3/9] Category Bar Chart Saved: {c3.name}")

    c4 = charts.plot_top_products(top_prods, CHARTS_DIR / "top_products.png")
    print(f"  ✓ [4/9] Top 10 Products Horizontal Bar Saved: {c4.name}")

    c5 = charts.plot_top_customers(top_custs, CHARTS_DIR / "top_customers.png")
    print(f"  ✓ [5/9] Top 10 Customers Horizontal Bar Saved: {c5.name}")

    c6 = charts.plot_quantity_distribution(qty_dist, CHARTS_DIR / "quantity_distribution.png")
    print(f"  ✓ [6/9] Quantity Distribution Histogram Saved: {c6.name}")

    c7 = charts.plot_revenue_vs_profit(rev_series, prof_series, corr_r, CHARTS_DIR / "revenue_profit.png")
    print(f"  ✓ [7/9] Revenue vs Profit Scatter Plot Saved: {c7.name}")

    c8 = charts.plot_category_share(cat_share, CHARTS_DIR / "category_share.png")
    print(f"  ✓ [8/9] Category Share Pie Chart Saved: {c8.name}")

    c9 = charts.plot_monthly_rolling(monthly_rolling, CHARTS_DIR / "monthly_revenue_rolling.png")
    print(f"  ✓ [9/9] Monthly Rolling Average Line Chart Saved: {c9.name}")

    # 4. Generate report
    print("\n[Phase 4/4] Compiling Executive Visualization Report...")
    rep = generate_visualization_report(df, REPORT_PATH)
    print(f"  ✓ Report written to: {rep.name}")

    print("\n" + "=" * 70)
    print("  🎉 PIPELINE COMPLETE: All 9 visualizations and executive report generated!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
