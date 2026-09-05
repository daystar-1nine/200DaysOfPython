"""
Task 3: Horizontal Bar Chart — Regional Revenue Comparison
Demonstrates the creation of a horizontal bar chart (barh) for regional comparison.
Shows how horizontal bars make label reading effortless and facilitate top-down ranking.
"""

# What is used: Non-interactive backend selection
# Why it is used: Ensures automated test and script execution without GUI popups
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib
# Why it is used: Factory interface for Figure and Axes
# How it works: Provides subplots() to create object-oriented chart canvases
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_horizontal_bar_chart(output_dir: Path) -> Path:
    """Creates and saves a regional revenue horizontal bar chart."""
    # Data definition
    regions = ["North", "South", "East", "West"]
    revenue = [420000, 380000, 310000, 540000]

    # What is used: pandas Series sorting
    # Why it is used: Ascending sort places the highest-performing category at the top of horizontal plots
    # How it works: Creates a Series and sorts ascending so top category is at the top of y-axis
    s = pd.Series(revenue, index=regions).sort_values(ascending=True)

    # What is used: plt.subplots()
    # Why it is used: Object-oriented canvas setup
    # How it works: Allocates 9x5 inch canvas
    fig, ax = plt.subplots(figsize=(9, 5))

    # What is used: ax.barh()
    # Why it is used: Renders horizontal bars extending along the x-axis
    # How it works: Takes category y-positions and quantitative widths
    colors = ["#7fa9c9", "#568db8", "#3370a6", "#175591"]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor="#0c3257", height=0.55)

    # What is used: ax.bar_label()
    # Why it is used: Directly displays formatted revenue strings beside each horizontal bar
    # How it works: Computes coordinates at bar endpoints and renders text
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=6, fontsize=10, fontweight="bold")

    # Set xlim headroom for labels
    ax.set_xlim(0, max(s.values) * 1.18)

    # What is used: ax.set_title(), ax.set_xlabel(), ax.set_ylabel()
    # Why it is used: Complete context and unit labeling
    # How it works: Assigns title, x-axis label, and y-axis label
    ax.set_title("Total Revenue by Geographic Region (H1 2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Total Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sales Region", fontsize=11, fontweight="bold")

    # Format x-axis with gridlines
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task3_horizontal_bar.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_horizontal_bar_chart(out_dir)
    print(f"[SUCCESS] Task 3 Horizontal Bar Chart saved to: {saved}")
