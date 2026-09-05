"""
Distribution Visualization Module
=================================
Generates univariate distribution charts, box plots, violin plots,
and observation-level strip/swarm plots using Seaborn.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_order_value_distribution(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 1: Histplot + KDE with mean/median reference lines.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Revenue",
        kde=True,
        bins=30,
        color="#1f77b4",
        edgecolor="white",
        stat="density",
        ax=ax
    )

    mean_val = df["Revenue"].mean()
    med_val = df["Revenue"].median()
    skew_val = df["Revenue"].skew()

    ax.axvline(mean_val, color="crimson", linestyle="--", linewidth=2.0, label=f"Mean: ₹{mean_val:,.0f}")
    ax.axvline(med_val, color="forestgreen", linestyle="-.", linewidth=2.0, label=f"Median: ₹{med_val:,.0f}")

    ax.annotate(
        f"Right-Skewed Distribution\nSkewness: {skew_val:.2f}\n(Mean > Median)",
        xy=(mean_val, ax.get_ylim()[1] * 0.6),
        xytext=(mean_val + 20000, ax.get_ylim()[1] * 0.75),
        arrowprops=dict(facecolor="#333333", arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", edgecolor="#fbc02d")
    )

    ax.xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Fig 1: Order Revenue Distribution (Histogram + Gaussian KDE)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_revenue_by_category_boxplot(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 2: Revenue across Product Categories via Boxplot with outlier isolation.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=df,
        x="Category",
        y="Revenue",
        hue="Category",
        legend=False,
        palette="Set2",
        ax=ax,
        flierprops=dict(marker="D", markersize=5, markerfacecolor="crimson", markeredgecolor="darkred", alpha=0.75),
        medianprops=dict(color="black", linewidth=2.0)
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Fig 2: Order Revenue by Product Category (IQR Outlier Analysis)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_profit_margin_violin(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 3: Profit Margin density across Customer Segments via Violin Plot.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(
        data=df,
        x="Customer_Segment",
        y="Profit_Margin",
        hue="Customer_Segment",
        legend=False,
        palette="muted",
        inner="quartile",
        cut=0,
        ax=ax
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
    ax.set_title("Fig 3: Profit Margin Distribution by Customer Segment (Violin Density)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Customer Segment", fontsize=11, fontweight="bold")
    ax.set_ylabel("Profit Margin (%)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_segment_strip_swarm(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 11: Granular observation scatter with Strip and Box overlay.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Boxplot underlay
    sns.boxplot(
        data=df,
        x="Customer_Segment",
        y="Revenue",
        color="lightgray",
        boxprops=dict(alpha=0.4),
        showfliers=False,
        ax=ax
    )

    # Stripplot overlay
    sns.stripplot(
        data=df,
        x="Customer_Segment",
        y="Revenue",
        hue="Customer_Segment",
        legend=False,
        palette="deep",
        jitter=0.25,
        size=5,
        alpha=0.7,
        ax=ax
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Fig 11: Customer Segment Revenue (Strip Plot with Boxplot Underlay)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Customer Segment", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
