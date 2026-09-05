"""
Task 6: Asymmetric Layouts with matplotlib.gridspec.GridSpec
Demonstrates the creation of an executive dashboard layout with 1 large prominent
time-series chart across the top and 3 smaller analytical subplots below.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless compatibility
# How it works: Switches to "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path


def create_gridspec_dashboard(output_dir: Path) -> Path:
    """Generates a 1-large + 3-small asymmetric layout using GridSpec."""
    # Data definitions
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [120000, 135000, 128000, 145000, 160000, 155000, 172000, 185000, 178000, 205000, 240000, 290000]
    regions = ["North", "South", "East", "West"]
    reg_rev = [480000, 420000, 360000, 620000]
    categories = ["Tech", "Home", "Apparel"]
    cat_shares = [52, 28, 20]
    np.random.seed(42)
    order_sizes = np.random.geometric(p=0.35, size=150)

    # What is used: plt.figure() and GridSpec(2, 3, height_ratios=[2.2, 1.5])
    # Why it is used: Implements asymmetric layout geometry not possible with rigid uniform subplots
    # How it works: Allocates 2 rows and 3 columns with custom height proportions and spacing
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.2, 1.5], hspace=0.32, wspace=0.25)

    # What is used: fig.add_subplot(gs[0, :])
    # Why it is used: Spans the large primary chart across all 3 columns of row 0
    # How it works: Slices entire row 0 as a single wide Axes
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(months, revenue, marker="o", color="#1f77b4", linewidth=2.5, label="Actual Monthly Revenue")
    ax_main.set_title("Executive Master Trend: Annual Revenue Progression (2026)", fontsize=14, fontweight="bold", pad=10)
    ax_main.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax_main.grid(True, linestyle="--", alpha=0.5)
    ax_main.legend(loc="upper left")

    # Bottom Subplot 1: Regional Revenue
    ax_sub1 = fig.add_subplot(gs[1, 0])
    bars = ax_sub1.bar(regions, reg_rev, color="#3182bd", edgecolor="#08519c", width=0.5)
    ax_sub1.set_title("Regional Distribution", fontsize=11, fontweight="bold")
    ax_sub1.set_ylabel("Revenue (₹)", fontsize=9.5)
    ax_sub1.grid(axis="y", linestyle=":", alpha=0.5)

    # Bottom Subplot 2: Category Shares (Pie)
    ax_sub2 = fig.add_subplot(gs[1, 1])
    ax_sub2.pie(cat_shares, labels=categories, autopct="%1.0f%%", startangle=140, colors=["#2ca02c", "#ff7f0e", "#9467bd"])
    ax_sub2.set_title("Category Share Breakdown", fontsize=11, fontweight="bold")

    # Bottom Subplot 3: Order Size Histogram
    ax_sub3 = fig.add_subplot(gs[1, 2])
    ax_sub3.hist(order_sizes, bins=np.arange(0.5, 10.5, 1), color="#e6550d", edgecolor="#7f2704", rwidth=0.8)
    ax_sub3.set_title("Order Size Frequency", fontsize=11, fontweight="bold")
    ax_sub3.set_xlabel("Units per Order", fontsize=9.5)
    ax_sub3.set_ylabel("Orders", fontsize=9.5)
    ax_sub3.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Asymmetric GridSpec Corporate Analytics Dashboard", fontsize=16, fontweight="bold", y=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task6_gridspec_layout.png"

    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_gridspec_dashboard(out)
    print(f"[SUCCESS] Task 6 saved to: {saved}")