"""
Challenge 1: 12-Month Sales vs Profit Comparative Line Chart
Visualizes monthly Sales and Profit trajectories across all 12 calendar months.
Demonstrates multi-series plotting, dual markers, customized styling, and financial annotations.
"""

# What is used: Non-interactive backend selection
# Why it is used: Ensures rendering works across headless testing environments
# How it works: Switches Matplotlib backend to "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib
# Why it is used: Factory interface for Figure and Axes
# How it works: Provides subplots() to create object-oriented chart canvases
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_sales_profit_trend(output_dir: Path) -> Path:
    """Creates a 12-month comparative line chart of Sales and Profit."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sales = [120000, 135000, 128000, 142000, 155000, 168000, 162000, 175000, 189000, 205000, 240000, 280000]
    profit = [26000, 31000, 28000, 33000, 37000, 42000, 39000, 44000, 49000, 56000, 68000, 82000]

    # What is used: plt.subplots()
    # Why it is used: Allocates OO canvas
    # How it works: Creates a 11x6 inch canvas
    fig, ax = plt.subplots(figsize=(11, 6))

    # What is used: ax.plot() for dual series
    # Why it is used: Compares two related time series on the same continuous horizontal timeline
    # How it works: Draws Sales in bold blue with circles and Profit in green with squares
    ax.plot(months, sales, color="#1f77b4", linewidth=2.5, marker="o", markersize=6, label="Gross Sales (₹)")
    ax.plot(months, profit, color="#2ca02c", linewidth=2.5, marker="s", markersize=6, linestyle="--", label="Net Profit (₹)")

    # Formatting
    ax.set_title("Full-Year Corporate Trajectory: Sales vs Net Profit (2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Financial Amount (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    # Annotate Q4 Holiday Peak
    ax.annotate(
        f"Year-End Peak:\nSales ₹{sales[-1]:,}\nProfit ₹{profit[-1]:,}",
        xy=(months[-1], sales[-1]),
        xytext=(-80, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc")
    )

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge1_sales_profit_line.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_sales_profit_trend(out_dir)
    print(f"[SUCCESS] Challenge 1 Chart saved to: {saved}")
