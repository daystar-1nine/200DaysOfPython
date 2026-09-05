"""
Challenge 5: Reusable Executive Sales Dashboard Generator
Implements an end-to-end reusable function create_sales_dashboard(df, output_path)
that automates KPI calculation, layout structuring, multi-chart rendering,
peak annotation, high-res saving, and memory closure.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless execution
# How it works: Sets "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import pandas as pd
from pathlib import Path


def create_sales_dashboard(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Automated reusable function to calculate KPIs and render an executive sales dashboard.

    # What is used: pandas aggregations, GridSpec, and Matplotlib OO API
    # Why it is used: Single-call production utility for recurring business intelligence reporting
    # How it works:
        1. Calculates core KPIs (Revenue, Profit, Orders, Margin)
        2. Configures asymmetric GridSpec layout
        3. Plots KPIs, Macro-Trend, and Category/Regional breakdowns
        4. Injects dynamic peak annotation
        5. Saves high-res PNG and closes figure
    """
    # 1. Calculate KPIs
    tot_rev = df["Revenue"].sum()
    tot_profit = df["Profit"].sum()
    tot_orders = df["Order_ID"].nunique() if "Order_ID" in df.columns else len(df)
    margin_pct = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0.0

    # 2. Aggregations
    monthly = df.groupby("Year_Month")["Revenue"].sum().reset_index() if "Year_Month" in df.columns else df.groupby("Month")["Revenue"].sum().reset_index()
    monthly_col = "Year_Month" if "Year_Month" in monthly.columns else "Month"
    reg_rev = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False) if "Region" in df.columns else pd.Series()
    cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False) if "Category" in df.columns else pd.Series()

    # 3. Create GridSpec Layout (3 rows: Row 0 KPIs, Row 1 Trend, Row 2 Breakdowns)
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 2.4, 2.2], hspace=0.35, wspace=0.25)

    # 4. Row 0: KPI Summary Header (Spanning both cols)
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis("off")
    kpi_text = (
        f"TOTAL REVENUE: ₹{tot_rev:,.0f}    |    "
        f"NET PROFIT: ₹{tot_profit:,.0f}    |    "
        f"TOTAL ORDERS: {tot_orders:,}    |    "
        f"PROFIT MARGIN: {margin_pct:.1f}%"
    )
    ax_kpi.text(
        0.5, 0.5,
        kpi_text,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1c3d5a",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef4fa", edgecolor="#b0c4de", lw=2.0)
    )

    # 5. Row 1: Macro Monthly Trend (Spanning both cols)
    ax_trend = fig.add_subplot(gs[1, :])
    ax_trend.plot(monthly[monthly_col].astype(str), monthly["Revenue"], marker="o", color="#1f77b4", linewidth=2.5, label="Monthly Revenue")
    avg_m_rev = monthly["Revenue"].mean()
    ax_trend.axhline(avg_m_rev, color="crimson", linestyle="--", linewidth=1.5, label=f"Average: ₹{avg_m_rev:,.0f}")

    # Dynamic Peak Annotation
    max_idx = monthly["Revenue"].idxmax()
    p_x = monthly[monthly_col].iloc[max_idx]
    p_y = monthly["Revenue"].iloc[max_idx]
    ax_trend.annotate(
        f"Peak: ₹{p_y:,.0f}",
        (str(p_x), p_y),
        xytext=(-30, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontweight="bold"
    )
    ax_trend.set_title("Executive Revenue Momentum & Benchmark", fontsize=13, fontweight="bold")
    ax_trend.set_ylabel("Revenue (₹)", fontsize=10, fontweight="bold")
    ax_trend.tick_params(axis="x", rotation=25, labelsize=9)
    ax_trend.grid(True, linestyle=":", alpha=0.5)
    ax_trend.legend(loc="upper left")

    # 6. Row 2 Left: Regional Revenue
    ax_reg = fig.add_subplot(gs[2, 0])
    if not reg_rev.empty:
        bars = ax_reg.bar(reg_rev.index, reg_rev.values, color="#3182bd", edgecolor="#08519c", width=0.55)
        ax_reg.bar_label(bars, fmt="₹{:,.0f}", padding=3, fontsize=8.5, fontweight="bold")
        ax_reg.set_ylim(0, max(reg_rev.values) * 1.15)
        ax_reg.set_title("Revenue by Geographic Region", fontsize=11, fontweight="bold")
        ax_reg.grid(axis="y", linestyle=":", alpha=0.5)

    # 7. Row 2 Right: Category Revenue
    ax_cat = fig.add_subplot(gs[2, 1])
    if not cat_rev.empty:
        bars_c = ax_cat.bar(cat_rev.index, cat_rev.values, color="#31a354", edgecolor="#006d2c", width=0.55)
        ax_cat.bar_label(bars_c, fmt="₹{:,.0f}", padding=3, fontsize=8.5, fontweight="bold")
        ax_cat.set_ylim(0, max(cat_rev.values) * 1.15)
        ax_cat.set_title("Revenue by Product Category", fontsize=11, fontweight="bold")
        ax_cat.tick_params(axis="x", rotation=15, labelsize=9)
        ax_cat.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("E-Commerce Performance Management Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # 8. Save & Close
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.loader import load_ecommerce_data
    from app.config import ECOMMERCE_CSV_PATH
    out = Path(__file__).resolve().parent.parent / "output" / "charts" / "challenge5_sales_dashboard.png"
    df = load_ecommerce_data(ECOMMERCE_CSV_PATH)
    saved = create_sales_dashboard(df, out)
    print(f"[SUCCESS] Challenge 5 saved to: {saved}")