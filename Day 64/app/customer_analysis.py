"""
Customer Analytics & Rankings Visualizations Module
===================================================
Visualizes top 10 customers by revenue and profit via horizontal bar charts.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_top_customers(top_rev_df: pd.DataFrame, top_prof_df: pd.DataFrame, output_path: str):
    """
    Chart 17: Side-by-side horizontal bar chart of Top 10 Customers by Revenue and Profit.
    """
    sns.set_theme(style=SNS_THEME)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Top by Revenue
    sns.barplot(
        data=top_rev_df,
        y="Customer_Name",
        x="Total_Revenue",
        hue="Customer_Name",
        legend=False,
        palette="Blues_r",
        ax=axes[0]
    )
    axes[0].set_title("Top 10 Enterprise Accounts by Revenue", fontsize=12, fontweight="bold")
    axes[0].xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-5:.1f}L")
    axes[0].set_xlabel("Cumulative Revenue (Rs.)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Customer Name", fontsize=10, fontweight="bold")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="Rs. %.0f", padding=3, fontsize=8)

    # Panel 2: Top by Profit
    sns.barplot(
        data=top_prof_df,
        y="Customer_Name",
        x="Total_Profit",
        hue="Customer_Name",
        legend=False,
        palette="Greens_r",
        ax=axes[1]
    )
    axes[1].set_title("Top 10 Enterprise Accounts by Profit", fontsize=12, fontweight="bold")
    axes[1].xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-5:.1f}L")
    axes[1].set_xlabel("Cumulative Profit (Rs.)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("")
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="Rs. %.0f", padding=3, fontsize=8)

    fig.suptitle("Chart 17: Top 10 Commercial Clients by Lifetime Financial Value", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
