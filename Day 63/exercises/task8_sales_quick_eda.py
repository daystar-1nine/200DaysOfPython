"""
Day 63 - Exercise 8: Integrated 4-Panel Quick Statistical EDA Dashboard
======================================================================
Demonstrates combining Matplotlib subplots with Seaborn axes-level functions
(histplot, barplot, scatterplot, heatmap) into a cohesive analytical view.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_task8():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid", palette="deep")

    # Allocate 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Revenue Distribution with KDE
    sns.histplot(df["Revenue"], kde=True, bins=25, color="#1f77b4", ax=axes[0, 0])
    axes[0, 0].set_title("1. Revenue Distribution (KDE Overlay)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Revenue (₹)")
    axes[0, 0].xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # Panel 2: Regional Revenue Mean with Bootstrapped CI
    sns.barplot(data=df, x="Region", y="Revenue", estimator="mean", errorbar=("ci", 95), palette="Set2", ax=axes[0, 1])
    axes[0, 1].set_title("2. Mean Regional Revenue (95% CI)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Mean Revenue (₹)")
    axes[0, 1].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # Panel 3: Revenue vs Profit Scatter
    sns.scatterplot(data=df, x="Revenue", y="Profit", hue="Category", alpha=0.7, ax=axes[1, 0])
    axes[1, 0].set_title("3. Revenue vs Profit by Category", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Revenue (₹)")
    axes[1, 0].set_ylabel("Profit (₹)")
    axes[1, 0].xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    axes[1, 0].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # Panel 4: Correlation Heatmap
    num_cols = ["Quantity", "Discount", "Revenue", "Profit", "Profit_Margin"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, ax=axes[1, 1])
    axes[1, 1].set_title("4. Key Metrics Correlation Matrix", fontsize=12, fontweight="bold")

    fig.suptitle("E-Commerce Quick Statistical EDA Multi-Panel Dashboard", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "exercise_task8_quick_eda.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 8 saved to: {out_file}")

if __name__ == "__main__":
    run_task8()
