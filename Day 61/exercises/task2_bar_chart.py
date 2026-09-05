"""
Task 2: Bar Chart — Product Sales Comparison
Demonstrates the creation of a vertical bar chart comparing categorical sales.
Includes sorting from highest to lowest, bar value annotations, and high-res export.
"""

# What is used: Non-interactive backend selection
# Why it is used: Ensures headless compatibility across CLI and automated pipelines
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib
# Why it is used: Core interface for Figure and Axes manipulation
# How it works: Provides subplots() to create object-oriented chart canvases
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_bar_chart(output_dir: Path) -> Path:
    """Creates and saves a sorted product sales vertical bar chart."""
    # Data definition
    products = ["Laptop Pro", "Noise-Cancelling Headphones", "Smartphone Ultra", "Mechanical Keyboard", "USB-C Dock"]
    sales = [850000, 320000, 940000, 180000, 140000]

    # What is used: pandas Series sorting
    # Why it is used: Sorting bars descending makes relative comparisons immediate and intuitive
    # How it works: Creates a Series and sorts values descending
    s = pd.Series(sales, index=products).sort_values(ascending=False)

    # What is used: plt.subplots()
    # Why it is used: Object-oriented canvas setup
    # How it works: Creates a canvas of 10x6 inches
    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: ax.bar()
    # Why it is used: Renders vertical rectangular bars proportional to metric magnitudes
    # How it works: Draws bars for categories with specified colors and edge outlines
    colors = ["#2b5c8f", "#3c7bb6", "#5698d4", "#7bb3e8", "#a6ccf5"]
    bars = ax.bar(s.index, s.values, color=colors, edgecolor="#1c3d5a", width=0.55)

    # What is used: ax.bar_label()
    # Why it is used: Places quantitative values directly over each bar for effortless reading
    # How it works: Computes coordinates of bar tops and positions currency-formatted text labels
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=4, fontsize=9, fontweight="bold")

    # Set headroom so labels do not collide with the upper axis spine
    ax.set_ylim(0, max(s.values) * 1.15)

    # What is used: ax.set_title(), ax.set_xlabel(), ax.set_ylabel()
    # Why it is used: Unambiguous communication of chart topic and units
    # How it works: Assigns title, x-axis label, and y-axis label
    ax.set_title("Top 5 Product Revenue Rankings (H1 2026)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Product", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Sales (₹)", fontsize=11, fontweight="bold")

    # Rotate x-axis labels slightly for readability
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task2_bar_chart.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_bar_chart(out_dir)
    print(f"[SUCCESS] Task 2 Bar Chart saved to: {saved}")
