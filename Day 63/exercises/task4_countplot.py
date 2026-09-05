"""
Day 63 - Exercise 4: Categorical Frequency Analysis with Countplot
=================================================================
Demonstrates frequency counting across categories, multi-group hue
subdivision, and automatic bar count labeling.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    # What is used: sns.countplot with hue="Region"
    # Why it is used: Compares order volumes across categories split by geographic territory
    # How it works: Counts occurrences of discrete rows matching Category x Region pairs
    sns.countplot(
        data=df,
        x="Category",
        hue="Region",
        palette="tab10",
        edgecolor="black",
        linewidth=0.6,
        ax=ax
    )

    # What is used: ax.bar_label across all container patches
    # Why it is used: Displays exact integer transaction volume directly above each bar
    # How it works: Iterates through bar containers and renders value labels
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=3)

    # Expand upper headroom to avoid label clipping
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title("Order Transaction Volume by Category & Region", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Orders", fontsize=11, fontweight="bold")
    ax.legend(title="Sales Region", loc="upper right", frameon=True)

    sns.despine(ax=ax, top=True, right=True)

    out_file = os.path.join(output_dir, "exercise_task4_countplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 4 saved to: {out_file}")

if __name__ == "__main__":
    run_task4()
