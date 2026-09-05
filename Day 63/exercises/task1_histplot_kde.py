"""
Day 63 - Exercise 1: Univariate Distribution Modeling with Histplot & KDE
========================================================================
Demonstrates density modeling, bin selection, kernel smoothing, and
statistical benchmark lines (Mean vs Median) using Seaborn histplot.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_task1():
    # What is used: Relative dataset path resolving and Pandas read_csv
    # Why it is used: Ensures platform-independent dataset loading
    # How it works: Locates the CSV relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # What is used: Seaborn set_theme with "whitegrid"
    # Why it is used: Establishes a clean, publication-quality aesthetic
    # How it works: Modifies Matplotlib rcParams with sensible statistical defaults
    sns.set_theme(style="whitegrid", palette="Blues_r")

    # What is used: Figure and single Axes allocation
    # Why it is used: Explicit object-oriented canvas control
    # How it works: Allocates a 10x6 inch figure container
    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: sns.histplot with kde=True and stat="density"
    # Why it is used: Visualizes empirical frequency distribution and smooth probability density simultaneously
    # How it works: Bins continuous data and overlays a Gaussian kernel density estimation curve
    sns.histplot(
        data=df,
        x="Revenue",
        kde=True,
        bins=30,
        color="#1f77b4",
        edgecolor="white",
        stat="density",
        ax=ax
    )

    # What is used: Statistical calculations (mean, median, skewness)
    # Why it is used: Quantifies distributional asymmetry and central tendency
    # How it works: Computes arithmetic average, 50th percentile, and 3rd standardized moment
    mean_val = df["Revenue"].mean()
    med_val = df["Revenue"].median()
    skew_val = df["Revenue"].skew()

    # What is used: ax.axvline for statistical reference lines
    # Why it is used: Highlights the divergence between Mean and Median caused by skewness
    # How it works: Draws vertical lines across the entire y-domain
    ax.axvline(mean_val, color="crimson", linestyle="--", linewidth=2.0, label=f"Mean: ₹{mean_val:,.0f}")
    ax.axvline(med_val, color="forestgreen", linestyle="-.", linewidth=2.0, label=f"Median: ₹{med_val:,.0f}")

    # What is used: ax.annotate for skewness callout
    # Why it is used: Directs executive attention to distributional characteristics
    # How it works: Places a stylized text box on the plot at specified coordinates
    ax.annotate(
        f"Right-Skewed\nSkewness: {skew_val:.2f}\n(Mean > Median)",
        xy=(mean_val, ax.get_ylim()[1] * 0.6),
        xytext=(mean_val + 25000, ax.get_ylim()[1] * 0.75),
        arrowprops=dict(facecolor="#333333", arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", edgecolor="#fbc02d")
    )

    ax.set_title("Order Revenue Distribution (Histogram + KDE Density)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)

    # What is used: sns.despine()
    # Why it is used: Removes unnecessary top and right spines, maximizing data-ink ratio
    # How it works: Hides specified border axes
    sns.despine(ax=ax, top=True, right=True)

    # What is used: fig.savefig and plt.close(fig)
    # Why it is used: Persists figure at 300 DPI and purges memory to prevent memory leaks
    # How it works: Encodes raster image to disk and unregisters canvas from pyplot state
    out_file = os.path.join(output_dir, "exercise_task1_histplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 1 saved to: {out_file}")

if __name__ == "__main__":
    run_task1()
