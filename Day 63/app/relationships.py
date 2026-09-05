"""
Relational & Multivariate Visualization Module
==============================================
Generates multi-dimensional scatterplots, linear regression models,
and aggregated time-series trends using Seaborn.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_order_value_vs_units_scatter(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 5: Multivariate scatterplot (Revenue vs Quantity with Segment Hue & Discount Size).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 7))

    sns.scatterplot(
        data=df,
        x="Quantity",
        y="Revenue",
        hue="Customer_Segment",
        size="Discount_Percent",
        sizes=(40, 250),
        alpha=0.75,
        palette="tab10",
        edgecolor="w",
        linewidth=0.8,
        ax=ax
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Fig 5: Order Revenue vs Quantity (Hue: Segment, Size: Discount %)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Quantity (Units Sold)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_order_value_vs_profit_regplot(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 10: Linear regression trendline of Revenue vs Profit with 95% confidence band.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(
        data=df,
        x="Revenue",
        y="Profit",
        scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 35},
        line_kws={"color": "darkred", "linewidth": 2.2, "label": "OLS Regression Fit"},
        ci=95,
        ax=ax
    )

    corr = df["Revenue"].corr(df["Profit"])
    ax.annotate(
        f"Pearson r = {corr:+.3f}\nStrong Linear Scalability",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#66bb6a")
    )

    ax.xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Fig 10: Order Revenue vs Net Profit (Linear Regression Fit with 95% CI)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Profit (₹)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_monthly_revenue_trend(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 9: Monthly revenue trendline partitioned by Category with variance bands.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=df,
        x="Month",
        y="Revenue",
        hue="Category",
        marker="o",
        errorbar=("ci", 95),
        palette="tab10",
        linewidth=2.0,
        ax=ax
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
    ax.set_title("Fig 9: Monthly Revenue Trajectory by Category (Mean & 95% CI Band)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Calendar Month (2026)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
