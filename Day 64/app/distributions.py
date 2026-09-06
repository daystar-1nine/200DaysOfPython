"""
Distribution Visualizations Module
==================================
Generates univariate distribution charts for Revenue, Profit, and Quantity.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_revenue_distribution(df: pd.DataFrame, output_path: str):
    """
    Chart 1: Revenue distribution (histplot + KDE) with mean and median markers.
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(df["Revenue"], kde=True, bins=30, color="#1f77b4", edgecolor="white", stat="density", ax=ax)
    mean_v = df["Revenue"].mean()
    med_v = df["Revenue"].median()

    ax.axvline(mean_v, color="crimson", linestyle="--", linewidth=2.0, label=f"Mean: Rs. {mean_v:,.0f}")
    ax.axvline(med_v, color="forestgreen", linestyle="-.", linewidth=2.0, label=f"Median: Rs. {med_v:,.0f}")

    ax.xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    ax.set_title("Chart 1: Order Revenue Distribution (Histogram + KDE Density)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Order Revenue (Rs.)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.legend()
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_profit_distribution(df: pd.DataFrame, output_path: str):
    """
    Chart 2: Profit distribution (histplot + KDE).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(df["Profit"], kde=True, bins=30, color="#2ca02c", edgecolor="white", stat="density", ax=ax)
    mean_v = df["Profit"].mean()
    med_v = df["Profit"].median()

    ax.axvline(mean_v, color="crimson", linestyle="--", linewidth=2.0, label=f"Mean: Rs. {mean_v:,.0f}")
    ax.axvline(med_v, color="black", linestyle="-.", linewidth=2.0, label=f"Median: Rs. {med_v:,.0f}")

    ax.xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    ax.set_title("Chart 2: Net Profit Distribution (Histogram + KDE Density)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Net Profit (Rs.)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.legend()
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_quantity_distribution(df: pd.DataFrame, output_path: str):
    """
    Chart 3: Quantity distribution (histplot).
    """
    sns.set_theme(style=SNS_THEME)
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.histplot(df["Quantity"], bins=15, discrete=True, color="#ff7f0e", edgecolor="black", ax=ax)
    ax.set_title("Chart 3: Order Quantity Distribution (Units Sold)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Quantity (Units)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Transaction Count", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
