"""
Task 1: Multiple Subplots (2x2 Layout)
Demonstrates creating a 2-row by 2-column multi-panel layout using plt.subplots().
Integrates Line, Bar, Scatter, and Histogram charts into a unified executive canvas.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Ensures headless rendering across CLI and automated testing
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def create_2x2_subplots(output_dir: Path) -> Path:
    """Generates a 2x2 multi-panel figure containing 4 distinct chart types."""
    # Data definitions
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    sales = [22000, 26000, 24000, 31000, 37000, 42000]
    categories = ["Tech", "Home", "Fashion", "Sports"]
    cat_rev = [450000, 320000, 210000, 180000]
    np.random.seed(42)
    units = np.random.randint(1, 20, 50)
    profit = units * 1200 + np.random.normal(0, 1500, 50)
    order_values = np.random.normal(3500, 800, 200)

    # What is used: plt.subplots(2, 2, figsize=(14, 10))
    # Why it is used: Allocates a 2x2 grid of Axes in an Object-Oriented paradigm
    # How it works: Returns canvas (fig) and a 2D numpy array of Axes (axes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Top-Left Line Chart (Trend)
    axes[0, 0].plot(months, sales, marker="o", color="#1f77b4", linewidth=2.2, label="Monthly Sales")
    axes[0, 0].set_title("1. Sales Growth Trend", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Sales (₹)", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    # Panel 2: Top-Right Bar Chart (Category Comparison)
    bars = axes[0, 1].bar(categories, cat_rev, color=["#2b5c8f", "#417db4", "#659ecf", "#8ebfec"], edgecolor="#111")
    axes[0, 1].bar_label(bars, fmt="₹{:,.0f}", padding=3, fontsize=9)
    axes[0, 1].set_title("2. Category Performance", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Revenue (₹)", fontsize=10)
    axes[0, 1].set_ylim(0, max(cat_rev) * 1.15)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 3: Bottom-Left Scatter Plot (Bivariate Association)
    axes[1, 0].scatter(units, profit, color="#d95f02", alpha=0.75, edgecolors="#662500")
    axes[1, 0].set_title("3. Units Sold vs Profit", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Units Purchased", fontsize=10)
    axes[1, 0].set_ylabel("Net Profit (₹)", fontsize=10)
    axes[1, 0].grid(True, linestyle=":", alpha=0.5)

    # Panel 4: Bottom-Right Histogram (Frequency Distribution)
    axes[1, 1].hist(order_values, bins=12, color="#7570b3", edgecolor="#1a1a1a", alpha=0.8)
    axes[1, 1].set_title("4. Order Value Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Order Value (₹)", fontsize=10)
    axes[1, 1].set_ylabel("Frequency", fontsize=10)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.5)

    # Global title and layout
    fig.suptitle("Executive 4-Panel Performance Overview", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task1_multiple_subplots.png"
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_2x2_subplots(out)
    print(f"[SUCCESS] Task 1 saved to: {saved}")