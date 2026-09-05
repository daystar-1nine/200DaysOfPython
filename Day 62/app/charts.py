"""
Modular Matplotlib Chart Components for Day 62.
Defines isolated plotting functions that accept an explicit Axes object,
enabling direct reuse between standalone figures and the master dashboard.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless compatibility across environments
# How it works: Activates "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pathlib import Path
from app.config import (
    DPI,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_DANGER,
    PALETTE_CATEGORICAL
)
from app.formatters import format_compact_inr, format_currency


def save_individual_chart(fig: plt.Figure, path: Path, dpi: int = DPI) -> Path:
    """Saves a standalone figure safely and releases canvas memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_kpi_card(ax: plt.Axes, title: str, value: str, subtitle: str, color: str = COLOR_PRIMARY) -> None:
    """Renders a polished, card-style executive metric tile on the given Axes."""
    ax.axis("off")
    content = f"{title.upper()}\n\n{value}\n\n{subtitle}"
    ax.text(
        0.5, 0.5,
        content,
        ha="center", va="center",
        fontsize=11, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8f9fa", edgecolor=color, lw=1.8)
    )


def plot_monthly_revenue_trend(ax: plt.Axes, data: pd.Series, avg_line: bool = True, annotate_max: bool = True) -> None:
    """Renders monthly revenue line trajectory with optional benchmark line and peak annotation."""
    ax.plot(
        data.index.astype(str),
        data.values,
        marker="o",
        color=COLOR_PRIMARY,
        linewidth=2.5,
        markersize=6,
        markerfacecolor=COLOR_SECONDARY,
        label="Monthly Revenue"
    )

    if avg_line:
        avg_val = data.mean()
        ax.axhline(
            avg_val,
            color=COLOR_DANGER,
            linestyle="--",
            linewidth=1.8,
            label=f"Avg Revenue ({format_compact_inr(avg_val)})"
        )

    if annotate_max and len(data) > 0:
        max_idx = data.values.argmax()
        m_x = str(data.index[max_idx])
        m_y = data.values[max_idx]
        ax.annotate(
            f"Peak: {format_compact_inr(m_y)}",
            xy=(m_x, m_y),
            xytext=(-35, 18),
            textcoords="offset points",
            arrowprops=dict(facecolor=COLOR_DANGER, edgecolor="#8b0000", arrowstyle="->", lw=1.5),
            fontweight="bold",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc")
        )

    ax.set_title("Executive Revenue Trajectory & Benchmark Target", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Revenue (₹)", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)


def plot_regional_revenue(ax: plt.Axes, data: pd.Series) -> None:
    """Renders regional revenue bar chart."""
    colors = ["#1b4f72", "#2e86c1", "#5dade2", "#aed6f1"][:len(data)]
    bars = ax.bar(data.index, data.values, color=colors, edgecolor="#111", width=0.55)
    ax.bar_label(bars, labels=[format_compact_inr(v) for v in data.values], padding=3, fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, data.values.max() * 1.15)
    ax.set_title("Geographic Performance by Region", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (₹)", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def plot_category_revenue(ax: plt.Axes, data: pd.Series) -> None:
    """Renders category revenue bar chart."""
    bars = ax.bar(data.index, data.values, color=PALETTE_CATEGORICAL[:len(data)], edgecolor="#111", width=0.55)
    ax.bar_label(bars, labels=[format_compact_inr(v) for v in data.values], padding=3, fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, data.values.max() * 1.15)
    ax.set_title("Revenue Contribution by Category", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (₹)", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def plot_top_products(ax: plt.Axes, data: pd.Series) -> None:
    """Renders horizontal bar chart for top N products."""
    bars = ax.barh(data.index, data.values, color="#3182bd", edgecolor="#08519c", height=0.6)
    ax.bar_label(bars, labels=[format_compact_inr(v) for v in data.values], padding=4, fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, data.values.max() * 1.20)
    ax.set_title("Top 10 High-Velocity Revenue SKUs", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Gross Sales (₹)", fontsize=9.5, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.grid(axis="x", linestyle="--", alpha=0.5)


def plot_revenue_vs_profit(ax: plt.Axes, revenue: pd.Series, profit: pd.Series, r: float) -> None:
    """Renders scatter plot of order-level revenue vs profit with trendline."""
    ax.scatter(revenue, profit, color="#fd8d3c", edgecolor="#a63603", alpha=0.6, s=45, label="Transactions")
    # Trendline
    slope, intercept = np.polyfit(revenue, profit, 1)
    x_vals = np.linspace(revenue.min(), revenue.max(), 100)
    ax.plot(x_vals, slope * x_vals + intercept, color="#08519c", linewidth=2.0, linestyle="--", label=f"Trendline (r={r:.2f})")
    ax.set_title("Order Revenue vs Profit Association", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Order Revenue (₹)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Net Profit (₹)", fontsize=9.5, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5)


def plot_quantity_distribution(ax: plt.Axes, data: pd.Series) -> None:
    """Renders order quantity frequency histogram."""
    counts, edges, patches = ax.hist(
        data,
        bins=np.arange(data.min() - 0.5, data.max() + 1.5, 1),
        color="#756bb1",
        edgecolor="#2b2353",
        alpha=0.85,
        rwidth=0.85
    )
    mean_q = data.mean()
    median_q = data.median()
    ax.axvline(mean_q, color=COLOR_DANGER, linestyle="--", linewidth=1.8, label=f"Mean: {mean_q:.1f}")
    ax.axvline(median_q, color=COLOR_SUCCESS, linestyle="-.", linewidth=1.8, label=f"Median: {median_q:.1f}")
    ax.set_title("Order Quantity Size Distribution", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Units Purchased", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Transaction Count", fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)


def plot_rolling_revenue(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Renders monthly revenue and 3-month rolling average line overlay."""
    col = "Year_Month" if "Year_Month" in df.columns else "Month"
    ax.plot(df[col].astype(str), df["Revenue"], color="#9ecae1", linewidth=1.8, marker="o", alpha=0.8, label="Actual Revenue")
    ax.plot(df[col].astype(str), df["Rolling_Avg"], color="#de2d26", linewidth=2.5, label="3M Moving Average")
    ax.fill_between(df[col].astype(str), df["Revenue"], df["Rolling_Avg"], color="#fee08b", alpha=0.4)
    ax.set_title("Revenue Momentum & 3-Month Moving Average", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (₹)", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_compact_inr(x)))
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5)