"""
Challenge 2: Top 5 Products Bar Chart
Identifies and visualizes the top 5 revenue-generating products from tabular data.
Emphasizes the #1 revenue leader using a distinct accent color and annotated value labels.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering in CLI environments
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib and pandas
# Why it is used: pandas aggregates and sorts products; pyplot renders the bars
# How it works: groupby -> sum -> sort_values -> ax.bar
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_top5_products_chart(output_dir: Path) -> Path:
    """Creates a bar chart showcasing the Top 5 products by revenue."""
    # Synthetic product portfolio
    data = {
        "Product": ["4K Gaming Monitor", "Wireless Ergonomic Mouse", "Mechanical Keyboard", "USB-C Hub Pro", "Noise-Cancelling Headphones", "Standing Desk", "Webcam 4K"],
        "Revenue": [650000, 180000, 320000, 140000, 520000, 480000, 210000]
    }
    df = pd.DataFrame(data)

    # What is used: pandas nlargest
    # Why it is used: Extracts exactly top 5 items sorted descending
    # How it works: Sorts DataFrame by Revenue descending and takes top 5 rows
    top5 = df.nlargest(5, "Revenue")

    # What is used: plt.subplots()
    # Why it is used: Instantiates figure and axes
    # How it works: Allocates a 10x6 inch canvas
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color highlighting: accent the #1 top product
    colors = ["#e41a1c" if i == 0 else "#377eb8" for i in range(len(top5))]

    # What is used: ax.bar()
    # Why it is used: Renders discrete vertical bars
    # How it works: Plots product names vs revenue
    bars = ax.bar(top5["Product"], top5["Revenue"], color=colors, edgecolor="#111111", width=0.55)

    # What is used: ax.bar_label()
    # Why it is used: Displays currency text directly above each bar
    # How it works: Computes positions of bar tops and renders formatted text
    ax.bar_label(bars, fmt="₹{:,.0f}", padding=4, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, top5["Revenue"].max() * 1.15)

    # Formatting
    ax.set_title("Top 5 Revenue-Generating Products (2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Product Name", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=12, labelsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge2_top5_products_bar.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_top5_products_chart(out_dir)
    print(f"[SUCCESS] Challenge 2 Chart saved to: {saved}")
