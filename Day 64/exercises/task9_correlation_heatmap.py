"""
Day 64 - Exercise 9: Correlation Matrix Heatmap & Extreme Pairs Identification
==============================================================================
Calculates Pearson correlation coefficients across numerical financial metrics
and visualizes the masked lower-triangle matrix with annotated extreme pairs.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_task9():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="white")

    cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Cost", "Profit"]
    corr = df[cols].corr()

    # What is used: np.triu mask
    # Why it is used: Hides upper symmetric duplicate cells and identity diagonal
    # How it works: Generates boolean matrix where True positions are masked
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7))

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
        linewidths=0.75,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax
    )

    ax.set_title("Task 9: Financial Features Correlation Heatmap", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    # Extract extreme non-diagonal pairs
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pos = pairs[0]
    top_neg = pairs[-1]

    print(f"Strongest Positive: {top_pos[0]} <-> {top_pos[1]} (r = {top_pos[2]:+.3f})")
    print(f"Strongest Negative: {top_neg[0]} <-> {top_neg[1]} (r = {top_neg[2]:+.3f})")

    out_file = os.path.join(output_dir, "exercise_task9_correlation_heatmap.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 9 saved to: {out_file}")

if __name__ == "__main__":
    run_task9()
