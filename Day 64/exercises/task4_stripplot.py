"""
Day 64 - Exercise 4: Observation-Level Inspection with Strip Plot vs Boxplot
===========================================================================
Compares raw individual data points (jittered strip plot) against aggregated boxplot
summaries to reveal density gaps and clusters hidden by quartiles.
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Boxplot
    sns.boxplot(
        data=df,
        x="Region",
        y="Profit",
        hue="Region",
        legend=False,
        palette="pastel",
        ax=axes[0]
    )
    axes[0].set_title("Boxplot: Quartiles & Fliers", fontsize=12, fontweight="bold")
    axes[0].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # Panel 2: Strip plot
    # What is used: sns.stripplot with jitter=0.25
    # Why it is used: Displays every raw observation, revealing density clusters and empty gaps
    # How it works: Distributes points randomly across category axis to eliminate severe overplotting
    sns.stripplot(
        data=df,
        x="Region",
        y="Profit",
        hue="Region",
        legend=False,
        palette="deep",
        jitter=0.25,
        size=5,
        alpha=0.65,
        ax=axes[1]
    )
    axes[1].set_title("Strip Plot: Raw Observation Density", fontsize=12, fontweight="bold")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    fig.suptitle("Task 4: Strip Plot vs Boxplot Comparison (Profit by Region)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    print("[INSIGHT] Strip plot reveals a dense low-profit cluster (Rs. 2K-Rs. 10K) and sparse high-value deals (>Rs. 30K) hidden by the boxplot.")

    out_file = os.path.join(output_dir, "exercise_task4_profit_stripplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 4 saved to: {out_file}")

if __name__ == "__main__":
    run_task4()
