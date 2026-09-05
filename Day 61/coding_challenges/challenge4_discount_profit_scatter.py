"""
Challenge 4: Discount vs Profit Scatter Plot & Correlation Audit
Investigates the bivariate relationship between discount percentage and net profit.
Computes Pearson correlation coefficient (r) using pandas and compares empirical data to visualization.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering in CLI and test environments
# How it works: Switches to "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot, numpy, and pandas modules
# Why it is used: pandas computes Pearson correlation; numpy fits trendline; pyplot renders scatter
# How it works: Loads data, calculates r, and visualizes points with trendline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def create_discount_profit_scatter(output_dir: Path) -> Path:
    """Creates a scatter plot comparing Discount vs Profit with correlation diagnostics."""
    # Deterministic simulation of orders
    np.random.seed(42)
    n_orders = 150
    discounts = np.random.uniform(0.0, 0.30, n_orders)
    # Profit inversely related to discount with natural variance
    profits = 15000 - (35000 * discounts) + np.random.normal(0, 2000, n_orders)

    df = pd.DataFrame({
        "Discount": discounts,
        "Profit": profits
    })

    # What is used: df["Discount"].corr(df["Profit"])
    # Why it is used: Calculates Pearson correlation coefficient between discount and profit
    # How it works: Normalizes covariance by product of standard deviations
    corr_val = df["Discount"].corr(df["Profit"])

    # What is used: plt.subplots()
    # Why it is used: Sets up OO canvas
    # How it works: Allocates 10x6 inch figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: ax.scatter()
    # Why it is used: Plots each order as a distinct point in Cartesian space
    # How it works: X is discount rate, Y is profit amount
    ax.scatter(
        df["Discount"] * 100,
        df["Profit"],
        color="#e41a1c",
        edgecolor="#4a0000",
        alpha=0.7,
        s=60,
        label="Transaction Orders"
    )

    # What is used: Trendline via np.polyfit
    # Why it is used: Visualizes negative trajectory across discount range
    # How it works: Fits slope and intercept to degree 1 polynomial
    slope, intercept = np.polyfit(df["Discount"] * 100, df["Profit"], 1)
    x_vals = np.linspace(0, 30, 100)
    ax.plot(x_vals, slope * x_vals + intercept, color="#377eb8", linewidth=2.2, linestyle="--", label=f"OLS Trendline (r = {corr_val:.3f})")

    # Formatting
    ax.set_title("Transaction Discount Rate vs Net Profit Margin", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Discount Applied (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Profit per Transaction (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Contextual annotation
    ax.text(
        0.05, 0.15,
        f"Negative Correlation (r = {corr_val:.3f}):\nAs discount percentages rise above 15%,\ntransactional margins erode sharply,\nconfirming heavy discounts degrade profitability.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff5f5", edgecolor="#e0a0a0")
    )

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge4_discount_profit_scatter.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_discount_profit_scatter(out_dir)
    print(f"[SUCCESS] Challenge 4 Scatter Plot saved to: {saved}")
