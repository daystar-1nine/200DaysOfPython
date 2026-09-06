"""
Categorical Visualizations Module
=================================
Generates boxplots, violins, barplots, and countplots.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_revenue_region_boxplot(df: pd.DataFrame, output_path: str):
    """
    Chart 4: Revenue by Region (boxplot).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Region",
        legend=False,
        palette="Blues_d",
        flierprops=dict(marker="D", markersize=5, markerfacecolor="red", alpha=0.7),
        ax=ax
    )
    ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    ax.set_title("Chart 4: Order Revenue Spread by Region (Boxplot & Outliers)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Revenue (Rs.)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_profit_category_boxplot(df: pd.DataFrame, output_path: str):
    """
    Chart 5: Profit by Category (boxplot).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=df,
        x="Category",
        y="Profit",
        hue="Category",
        legend=False,
        palette="Set2",
        flierprops=dict(marker="o", markersize=5, markerfacecolor="darkred", alpha=0.7),
        ax=ax
    )
    ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    ax.set_title("Chart 5: Net Profit Distribution by Product Category (Boxplot)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Profit (Rs.)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_quantity_category_violin(df: pd.DataFrame, output_path: str):
    """
    Chart 6: Quantity by Category (violinplot).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.violinplot(
        data=df,
        x="Category",
        y="Quantity",
        hue="Category",
        legend=False,
        palette="pastel",
        inner="quartile",
        cut=0,
        ax=ax
    )
    ax.set_title("Chart 6: Order Quantity Distribution by Category (Violin Plot)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Quantity (Units Sold)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_regional_revenue(regional_df: pd.DataFrame, output_path: str):
    """
    Chart 7: Average Revenue by Region (barplot from pre-calculated DataFrame).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=regional_df,
        x="Region",
        y="Mean_Revenue",
        hue="Region",
        legend=False,
        palette="Blues_d",
        edgecolor="black",
        ax=ax
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="Rs. %.0f", padding=3, fontsize=9)

    ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title("Chart 7: Average Order Revenue by Region", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Revenue (Rs.)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_category_profit(category_df: pd.DataFrame, output_path: str):
    """
    Chart 8: Average Profit by Category (barplot from pre-calculated DataFrame).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=category_df,
        x="Category",
        y="Mean_Profit",
        hue="Category",
        legend=False,
        palette="Greens_d",
        edgecolor="black",
        ax=ax
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="Rs. %.0f", padding=3, fontsize=9)

    ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title("Chart 8: Average Net Profit by Product Category", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Net Profit (Rs.)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_region_counts(counts_df: pd.DataFrame, output_path: str):
    """
    Chart 9: Transaction Count by Region (countplot/barplot of frequencies).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=counts_df,
        x="Region",
        y="Order_Count",
        hue="Region",
        legend=False,
        palette="tab10",
        edgecolor="black",
        ax=ax
    )
    for c in ax.containers:
        ax.bar_label(c, fontsize=10, padding=3, fontweight="bold")

    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title("Chart 9: Total Transaction Volume by Region", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Count", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
