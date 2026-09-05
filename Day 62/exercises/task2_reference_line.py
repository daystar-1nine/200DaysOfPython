"""
Task 2: Reference Line — Benchmarking Sales Against Annual Average
Demonstrates the use of axhline() to establish an executive performance threshold.
Enables viewers to instantly distinguish above-average and below-average months.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Ensures headless rendering across testing environments
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_reference_line_chart(output_dir: Path) -> Path:
    """Generates a monthly sales chart with an axhline() reference line."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sales = [42000, 46000, 39000, 51000, 58000, 54000, 62000, 67000, 59000, 72000, 81000, 89000]

    df = pd.DataFrame({"Month": months, "Sales": sales})
    avg_sales = df["Sales"].mean()

    # What is used: plt.subplots()
    # Why it is used: Object-oriented canvas setup
    # How it works: Creates 10x6 inch figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: Bar plot with dynamic color encoding based on threshold
    # Why it is used: Instantly visualizes whether a month beat or lagged the benchmark
    # How it works: Evaluates each value against average; assigns blue if above, gray if below
    bar_colors = ["#2b8cbe" if s >= avg_sales else "#a6bddb" for s in df["Sales"]]
    bars = ax.bar(df["Month"], df["Sales"], color=bar_colors, edgecolor="#1c4d6f", width=0.55)

    # What is used: ax.axhline()
    # Why it is used: Draws a constant horizontal reference benchmark across the canvas
    # How it works: Renders a line at y=avg_sales spanning from 0% to 100% of x-axis
    ax.axhline(
        avg_sales,
        color="#e41a1c",
        linestyle="--",
        linewidth=2.2,
        label=f"Annual Average Sales (₹{avg_sales:,.0f})"
    )

    # Add bar labels
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=3, fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, max(df["Sales"]) * 1.15)

    # Title & labels
    ax.set_title("Monthly Sales Performance vs Corporate Benchmark", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gross Sales (₹)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task2_reference_line.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_reference_line_chart(out)
    print(f"[SUCCESS] Task 2 saved to: {saved}")