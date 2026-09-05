"""
Challenge 3: Customer Spending Histogram (10 Bins)
Visualizes customer spending distribution across 10 discrete intervals.
Overlays Mean and Median vertical reference lines to analyze distribution skewness.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering in CLI and test environments
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot and numpy modules
# Why it is used: numpy simulates realistic log-normal customer spending; pyplot creates histogram
# How it works: Generates right-skewed spending data and plots 10-bin histogram
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_customer_spending_histogram(output_dir: Path) -> Path:
    """Generates a 10-bin histogram of customer spending distribution."""
    # Seed for deterministic reproducibility
    np.random.seed(101)

    # Simulate realistic customer spending (log-normal distribution: many small buyers, few whales)
    spending = np.random.lognormal(mean=8.5, sigma=0.6, size=250)

    mean_spend = np.mean(spending)
    median_spend = np.median(spending)

    # What is used: plt.subplots()
    # Why it is used: Sets up OO figure canvas
    # How it works: Allocates 10x6 inch figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: ax.hist() with bins=10
    # Why it is used: Visualizes frequency distribution partitioned into 10 contiguous bins
    # How it works: Computes 10 equal-width bins between min and max spending and counts observations
    counts, bin_edges, patches = ax.hist(
        spending,
        bins=10,
        color="#4575b4",
        edgecolor="#1b385a",
        alpha=0.8,
        linewidth=1.2,
        label="Customer Spend Frequency"
    )

    # What is used: ax.axvline()
    # Why it is used: Demonstrates positive skewness by showing Mean is pulled to the right of Median
    # How it works: Draws vertical line from y=0 to y=max at specified x value
    ax.axvline(mean_spend, color="#d73027", linestyle="--", linewidth=2.0, label=f"Mean Spend: ₹{mean_spend:,.0f}")
    ax.axvline(median_spend, color="#1a9850", linestyle="-.", linewidth=2.0, label=f"Median Spend: ₹{median_spend:,.0f}")

    # Formatting
    ax.set_title("Customer Annual Spending Distribution (N = 250 Customers)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Annual Spending (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Customers", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Annotation regarding right-skew
    ax.text(
        0.55, 0.45,
        f"Right-Skewed Distribution:\nMean (₹{mean_spend:,.0f}) > Median (₹{median_spend:,.0f})\nHigh-value VIP accounts pull\nthe arithmetic mean upward.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fefefe", edgecolor="#cccccc")
    )

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge3_customer_spending_hist.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_customer_spending_histogram(out_dir)
    print(f"[SUCCESS] Challenge 3 Histogram saved to: {saved}")
