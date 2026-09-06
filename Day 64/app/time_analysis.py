"""
Longitudinal Time-Series Visualizations Module
==============================================
Generates multi-series monthly revenue and profit trendlines with hue="Region".
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_monthly_revenue(df: pd.DataFrame, output_path: str):
    """
    Chart 14: Monthly Revenue (lineplot with hue="Region").
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    sns.lineplot(
        data=df,
        x="Month",
        y="Revenue",
        hue="Region",
        marker="o",
        linewidth=2.2,
        errorbar=None,
        estimator="sum",
        palette="tab10",
        ax=ax
    )

    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
    ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-5:.1f}L")
    ax.set_title("Chart 14: Monthly Aggregate Revenue Trajectory by Region", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Calendar Month (2026)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Monthly Revenue (Rs.)", fontsize=11, fontweight="bold")
    ax.legend(title="Sales Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_monthly_profit(df: pd.DataFrame, output_path: str):
    """
    Chart 15: Monthly Profit (lineplot with hue="Region").
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    sns.lineplot(
        data=df,
        x="Month",
        y="Profit",
        hue="Region",
        marker="s",
        linewidth=2.2,
        errorbar=None,
        estimator="sum",
        palette="tab10",
        ax=ax
    )

    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
    ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-5:.1f}L")
    ax.set_title("Chart 15: Monthly Aggregate Profit Trajectory by Region", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Calendar Month (2026)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Monthly Profit (Rs.)", fontsize=11, fontweight="bold")
    ax.legend(title="Sales Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
