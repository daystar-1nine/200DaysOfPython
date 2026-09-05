"""
Dedicated Matplotlib Visualization Engine.
Contains pure plotting functions using the Object-Oriented Figure and Axes paradigm.
Implements memory management via save_chart() and plt.close(fig).
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless chart generation across test suites and server processes
# How it works: Switches backend to "Agg" prior to importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from app.config import (
    DPI,
    FIGSIZE_STANDARD,
    FIGSIZE_WIDE,
    FIGSIZE_HORIZONTAL,
    FIGSIZE_PIE,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_DANGER,
    PALETTE_CATEGORICAL
)


def save_chart(fig: plt.Figure, path: Path, dpi: int = DPI) -> Path:
    """
    Saves a Matplotlib figure safely to disk and closes canvas memory.

    # What is used: fig.tight_layout(), fig.savefig(), and plt.close(fig)
    # Why it is used: Auto-scales margins, writes publication-grade raster images, and reclaims memory
    # How it works: Ensures parent dir exists, saves to path at specified DPI, and calls plt.close(fig)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_revenue(data: pd.Series, output_path: Path) -> Path:
    """Chart 1: Line Chart — Monthly Revenue Trend."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        data.index.astype(str),
        data.values,
        color=COLOR_PRIMARY,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor=COLOR_SECONDARY,
        label="Monthly Revenue"
    )
    ax.set_title("Monthly Revenue Trajectory (Year 2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month Period", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Revenue (₹)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    return save_chart(fig, output_path)


def plot_revenue_by_region(data: pd.Series, output_path: Path) -> Path:
    """Chart 2: Vertical Bar Chart — Revenue by Region."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"][:len(data)]
    bars = ax.bar(data.index, data.values, color=colors, edgecolor="#111111", width=0.5)
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=4, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, data.values.max() * 1.15)
    ax.set_title("Geographic Sales Distribution: Total Revenue by Region", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    return save_chart(fig, output_path)


def plot_revenue_by_category(data: pd.Series, output_path: Path) -> Path:
    """Chart 3: Vertical Bar Chart — Revenue by Category (Sorted Descending)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars = ax.bar(data.index, data.values, color=PALETTE_CATEGORICAL[:len(data)], edgecolor="#111111", width=0.55)
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=4, fontsize=9, fontweight="bold")
    ax.set_ylim(0, data.values.max() * 1.15)
    ax.set_title("Product Portfolio Breakdown: Revenue by Category", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15, labelsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    return save_chart(fig, output_path)


def plot_top_products(data: pd.Series, output_path: Path) -> Path:
    """Chart 4: Horizontal Bar Chart — Top 10 Products."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HORIZONTAL)
    bars = ax.barh(data.index, data.values, color="#3182bd", edgecolor="#08519c", height=0.6)
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=5, fontsize=9, fontweight="bold")
    ax.set_xlim(0, data.values.max() * 1.20)
    ax.set_title("Top 10 Revenue-Generating Products", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Total Sales Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Product Catalog Item", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    return save_chart(fig, output_path)


def plot_top_customers(data: pd.Series, output_path: Path) -> Path:
    """Chart 5: Horizontal Bar Chart — Top 10 Customers."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HORIZONTAL)
    bars = ax.barh(data.index, data.values, color="#31a354", edgecolor="#006d2c", height=0.6)
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=5, fontsize=9, fontweight="bold")
    ax.set_xlim(0, data.values.max() * 1.20)
    ax.set_title("Top 10 High-Value VIP Customer Accounts", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Total Lifetime Spend (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Customer Account Name", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    return save_chart(fig, output_path)


def plot_quantity_distribution(data: pd.Series, output_path: Path) -> Path:
    """Chart 6: Histogram — Order Quantity Distribution."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    counts, edges, patches = ax.hist(
        data,
        bins=np.arange(data.min() - 0.5, data.max() + 1.5, 1),
        color="#756bb1",
        edgecolor="#2b2353",
        alpha=0.85,
        rwidth=0.85
    )
    mean_qty = data.mean()
    median_qty = data.median()
    ax.axvline(mean_qty, color=COLOR_DANGER, linestyle="--", linewidth=2.0, label=f"Mean Qty: {mean_qty:.1f}")
    ax.axvline(median_qty, color=COLOR_SUCCESS, linestyle="-.", linewidth=2.0, label=f"Median Qty: {median_qty:.1f}")
    ax.set_title("Distribution of Order Quantities Across Transactions", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Units Purchased per Order", fontsize=11, fontweight="bold")
    ax.set_ylabel("Transaction Count (Frequency)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    return save_chart(fig, output_path)


def plot_revenue_vs_profit(revenue: pd.Series, profit: pd.Series, r: float, output_path: Path) -> Path:
    """Chart 7: Scatter Plot — Revenue vs Profit Relationship."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.scatter(
        revenue,
        profit,
        color="#fd8d3c",
        edgecolor="#a63603",
        alpha=0.65,
        s=50,
        label="Orders"
    )
    # Trendline
    slope, intercept = np.polyfit(revenue, profit, 1)
    x_vals = np.linspace(revenue.min(), revenue.max(), 100)
    ax.plot(x_vals, slope * x_vals + intercept, color="#08519c", linewidth=2.2, linestyle="--", label=f"OLS Linear Trend (r = {r:.3f})")
    ax.set_title("Order-Level Bivariate Relationship: Revenue vs Profit", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Order Net Profit (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    return save_chart(fig, output_path)


def plot_category_share(data: pd.Series, output_path: Path) -> Path:
    """Chart 8: Pie Chart — Category Revenue Share."""
    fig, ax = plt.subplots(figsize=FIGSIZE_PIE)
    explode = [0.06 if i == 0 else 0.0 for i in range(len(data))]
    wedges, texts, autotexts = ax.pie(
        data.values,
        labels=data.index,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
        colors=PALETTE_CATEGORICAL[:len(data)],
        shadow=True,
        textprops=dict(color="#1a1a1a", fontsize=10, fontweight="bold")
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9.5)
        autotext.set_weight("bold")
    ax.set_title("Relative Revenue Contribution by Category", fontsize=14, fontweight="bold", pad=14)
    return save_chart(fig, output_path)


def plot_monthly_rolling(data: pd.DataFrame, output_path: Path) -> Path:
    """Chart 9 (Bonus): Monthly Revenue with 3-Month Rolling Average Overlay."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        data["Year_Month"].astype(str),
        data["Revenue"],
        color="#9ecae1",
        linewidth=2.0,
        marker="o",
        markersize=6,
        alpha=0.85,
        label="Actual Monthly Revenue"
    )
    ax.plot(
        data["Year_Month"].astype(str),
        data["Rolling_Avg"],
        color="#de2d26",
        linewidth=2.8,
        label="3-Month Rolling Average"
    )
    ax.fill_between(
        data["Year_Month"].astype(str),
        data["Revenue"],
        data["Rolling_Avg"],
        color="#fee08b",
        alpha=0.4,
        label="Volatility Band"
    )
    ax.set_title("Monthly Revenue with 3-Month Rolling Trend Analysis", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month Period", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    return save_chart(fig, output_path)
