"""
Day 63 - Exercise 6: Correlation Matrix Heatmap with Upper Triangle Mask
========================================================================
Demonstrates computing Pearson correlation coefficients, masking symmetric
redundancies, and rendering an annotated diverging colormap heatmap.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_task6():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="white")

    # Select numerical columns
    numeric_cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Cost", "Profit", "Profit_Margin"]
    corr = df[numeric_cols].corr()

    # What is used: np.triu mask
    # Why it is used: Masks the upper symmetric triangle to remove visual noise and redundant duplication
    # How it works: Generates a boolean upper triangular matrix where True entries are masked out
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))

    # What is used: sns.heatmap with diverging colormap "coolwarm"
    # Why it is used: Clearly separates positive correlation (red) from negative correlation (blue)
    # How it works: Colors each cell according to Pearson r in [-1, +1]
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax
    )

    ax.set_title("E-Commerce Metrics Correlation Matrix (Masked Lower Triangle)", fontsize=13, fontweight="bold", pad=15)

    out_file = os.path.join(output_dir, "exercise_task6_heatmap.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 6 saved to: {out_file}")

if __name__ == "__main__":
    run_task6()
