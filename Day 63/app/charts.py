"""
Master Chart Orchestrator Module
================================
Coordinates execution of all 12 publication-quality statistical visualizations
and constructs the multi-panel Executive Dashboard.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.config import CHARTS_DIR, DEFAULT_DPI, SNS_THEME
from app.distributions import (
    plot_order_value_distribution,
    plot_revenue_by_category_boxplot,
    plot_profit_margin_violin,
    plot_segment_strip_swarm
)
from app.categorical import (
    plot_orders_by_category_and_segment,
    plot_regional_revenue_barplot
)
from app.relationships import (
    plot_order_value_vs_units_scatter,
    plot_order_value_vs_profit_regplot,
    plot_monthly_revenue_trend
)
from app.correlation import (
    plot_correlation_heatmap,
    plot_multivariate_pairplot
)

def generate_executive_dashboard(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 12: 4-Panel Executive Statistical Dashboard.
    Combines distribution, categorical comparison, bivariate scatter, and correlation.
    """
    sns.set_theme(style=SNS_THEME, palette="deep")
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # Panel 1: Revenue distribution with KDE
    sns.histplot(df["Revenue"], kde=True, bins=25, color="#1f77b4", ax=axes[0, 0])
    axes[0, 0].axvline(df["Revenue"].median(), color="crimson", linestyle="--", linewidth=1.8, label=f"Median: ₹{df['Revenue'].median()*1e-3:.0f}K")
    axes[0, 0].set_title("1. Revenue Density & Central Tendency", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Order Revenue (₹)")
    axes[0, 0].xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    axes[0, 0].legend()

    # Panel 2: Revenue across Categories (Boxplot)
    sns.boxplot(data=df, x="Category", y="Revenue", hue="Category", legend=False, palette="Set2", ax=axes[0, 1])
    axes[0, 1].set_title("2. Category Revenue Spread & Outliers", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Revenue (₹)")
    axes[0, 1].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    axes[0, 1].tick_params(axis="x", rotation=15)

    # Panel 3: Revenue vs Profit Regression
    sns.regplot(data=df, x="Revenue", y="Profit", scatter_kws={"alpha": 0.45, "color": "#2ca02c"}, line_kws={"color": "darkred", "linewidth": 2.2}, ax=axes[1, 0])
    axes[1, 0].set_title("3. Revenue vs Profit (OLS Linear Fit)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Revenue (₹)")
    axes[1, 0].set_ylabel("Profit (₹)")
    axes[1, 0].xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    axes[1, 0].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # Panel 4: Masked Correlation Matrix
    num_cols = ["Quantity", "Discount_Percent", "Revenue", "Cost", "Profit", "Profit_Margin"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, ax=axes[1, 1])
    axes[1, 1].set_title("4. Key Metrics Correlation Structure", fontsize=12, fontweight="bold")

    fig.suptitle("Executive Statistical EDA Multi-Panel Dashboard", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def generate_all_charts(df: pd.DataFrame, output_dir: str = None) -> list[str]:
    """
    Generates all 12 publication-grade figures.
    """
    out_dir = output_dir if output_dir is not None else CHARTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    manifest = [
        ("fig01_order_value_distribution.png", plot_order_value_distribution),
        ("fig02_revenue_by_category_boxplot.png", plot_revenue_by_category_boxplot),
        ("fig03_profit_margin_violin_by_segment.png", plot_profit_margin_violin),
        ("fig04_orders_by_category_and_segment_countplot.png", plot_orders_by_category_and_segment),
        ("fig05_order_value_vs_units_scatter.png", plot_order_value_vs_units_scatter),
        ("fig06_numerical_correlation_heatmap.png", plot_correlation_heatmap),
        ("fig07_multivariate_pairplot.png", plot_multivariate_pairplot),
        ("fig08_regional_revenue_barplot.png", plot_regional_revenue_barplot),
        ("fig09_monthly_revenue_trend_lineplot.png", plot_monthly_revenue_trend),
        ("fig10_order_value_vs_profit_regplot.png", plot_order_value_vs_profit_regplot),
        ("fig11_customer_segment_strip_swarm.png", plot_segment_strip_swarm),
        ("fig12_executive_statistical_dashboard.png", generate_executive_dashboard)
    ]

    generated = []
    for filename, fn in manifest:
        dest = os.path.join(out_dir, filename)
        fn(df, dest)
        generated.append(dest)
        print(f"Generated chart: {filename}")

    return generated
