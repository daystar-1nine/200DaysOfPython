"""
Categorical Visualization Module
================================
Generates category countplots and aggregated barplots with bootstrapped error bars.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_orders_by_category_and_segment(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 4: Transaction frequency by Category partitioned by Customer Segment.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.countplot(
        data=df,
        x="Category",
        hue="Customer_Segment",
        palette="deep",
        edgecolor="black",
        linewidth=0.6,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=3)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title("Fig 4: Transaction Volume by Product Category & Customer Segment", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Count", fontsize=11, fontweight="bold")
    ax.legend(title="Customer Segment", loc="upper right")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_regional_revenue_barplot(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 8: Mean Regional Revenue with 95% bootstrapped confidence intervals.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Region",
        legend=False,
        estimator="mean",
        errorbar=("ci", 95),
        palette="Blues_d",
        capsize=0.1,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="₹%.0f", fontsize=9, padding=4)

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title("Fig 8: Mean Regional Revenue (95% Bootstrapped Confidence Intervals)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Order Revenue (₹)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
