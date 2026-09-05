"""
Day 63 - Exercise 5: Multivariate Relational Analysis with Scatterplot & Hue
===========================================================================
Demonstrates multidimensional encoding: Revenue (X), Profit (Y),
Category (Hue), and Quantity (Size) with transparency tuning.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task5():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(11, 7))

    # What is used: sns.scatterplot with hue="Category" and size="Quantity"
    # Why it is used: Encodes 4 dimensions into a single readable 2D visualization
    # How it works: Maps position to financial metrics, color to product category, and point diameter to order volume
    sns.scatterplot(
        data=df,
        x="Revenue",
        y="Profit",
        hue="Category",
        size="Quantity",
        sizes=(30, 250),
        alpha=0.75,
        palette="deep",
        edgecolor="w",
        linewidth=0.8,
        ax=ax
    )

    ax.xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    ax.set_title("Order Revenue vs Net Profit (Hue: Category, Size: Quantity)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Profit (₹)", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    sns.despine(ax=ax, top=True, right=True)

    out_file = os.path.join(output_dir, "exercise_task5_scatterplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 5 saved to: {out_file}")

if __name__ == "__main__":
    run_task5()
