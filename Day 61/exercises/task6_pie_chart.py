"""
Task 6: Pie Chart — Category Revenue Share vs Bar Chart Comparative Discussion
Demonstrates creating a pie chart for proportional breakdown and critically evaluating
when a bar chart offers superior visual cognition.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering across CLI environments
# How it works: Sets backend to "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib
# Why it is used: Factory interface for Figure and Axes
# How it works: Provides subplots() to create object-oriented chart canvases
import matplotlib.pyplot as plt
from pathlib import Path


def create_pie_chart(output_dir: Path) -> Path:
    """Creates a pie chart of category revenue share with explanatory commentary."""
    categories = ["Electronics", "Furniture", "Apparel", "Kitchenware", "Fitness"]
    revenue = [450000, 320000, 240000, 180000, 110000]
    total_rev = sum(revenue)

    # What is used: plt.subplots()
    # Why it is used: Creates standard OO canvas
    # How it works: Allocates 8x7 inch canvas
    fig, ax = plt.subplots(figsize=(8, 7))

    # Slice highlighting: explode largest category
    explode = (0.08, 0, 0, 0, 0)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # What is used: ax.pie()
    # Why it is used: Visualizes part-to-whole proportions totaling 100%
    # How it works: Calculates angular slice dimensions proportional to values; applies percentage labels
    wedges, texts, autotexts = ax.pie(
        revenue,
        labels=categories,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
        colors=colors,
        shadow=True,
        textprops=dict(color="#1a1a1a", fontsize=10, fontweight="bold")
    )

    # Customize percentage text formatting
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9.5)
        autotext.set_weight("bold")

    ax.set_title("Product Category Revenue Distribution (H1 2026)", fontsize=13, fontweight="bold", pad=16)

    # Add analytical commentary answering: "Would a bar chart communicate this comparison better?"
    fig.text(
        0.5, 0.03,
        "Analytical Evaluation: While this pie chart shows broad proportions well for 5 categories,\n"
        "a horizontal bar chart would communicate exact numerical differences (e.g. ₹4.5L vs ₹3.2L)\n"
        "with far greater cognitive precision and zero angular estimation error.",
        ha="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbee", edgecolor="#e0d0a0")
    )

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task6_pie_chart.png"

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_pie_chart(out_dir)
    print(f"[SUCCESS] Task 6 Pie Chart saved to: {saved}")
