"""
Day 63 - Exercise 3: Density Comparison & Multimodality with Violin Plots
=======================================================================
Demonstrates kernel density estimation combined with inner quartiles
to compare regional revenue distributions and detect bimodal clustering.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: sns.violinplot with inner="quartile"
    # Why it is used: Combines the box plot quartiles with full density shapes, uncovering multimodal peaks
    # How it works: Fits Gaussian KDE along each region category and mirrors the shape
    sns.violinplot(
        data=df,
        x="Region",
        y="Revenue",
        palette="muted",
        inner="quartile",
        cut=0,  # Prevents density curve extending below zero revenue
        ax=ax
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Regional Revenue Distribution (Violin Density & Quartiles)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")

    sns.despine(ax=ax, top=True, right=True)

    out_file = os.path.join(output_dir, "exercise_task3_violinplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 3 saved to: {out_file}")

if __name__ == "__main__":
    run_task3()
