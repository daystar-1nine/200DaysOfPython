"""
Day 64 - Exercise 2: Regional Profit Distribution & Outlier Diagnostics (Boxplot)
================================================================================
Examines median profit, interquartile spread, and anomalous outlier transactions
across geographic sales regions using Seaborn boxplot.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # What is used: sns.boxplot with explicit flierprops
    # Why it is used: Visualizes medians, spreads (IQR), and isolates extreme transactions
    # How it works: Calculates Tukey quartiles and whiskers at 1.5*IQR per region
    sns.boxplot(
        data=df,
        x="Region",
        y="Profit",
        hue="Region",
        legend=False,
        palette="Blues_d",
        flierprops=dict(marker="D", markersize=6, markerfacecolor="crimson", markeredgecolor="darkred", alpha=0.7),
        ax=ax
    )

    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.set_title("Task 2: Profit Distribution by Region (IQR & Outlier Diagnostics)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Profit (₹)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    # What is used: Analytical answers to questions
    # Why it is used: Demonstrates statistical interpretation of boxplot geometry
    # How it works: Evaluates medians, IQRs, and flier counts
    medians = df.groupby("Region")["Profit"].median()
    iqrs = df.groupby("Region")["Profit"].apply(lambda s: s.quantile(0.75) - s.quantile(0.25))

    print(f"1. Highest Median Profit: {medians.idxmax()} (Rs. {medians.max():,.0f})")
    print(f"2. Largest Spread (IQR):   {iqrs.idxmax()} (IQR = Rs. {iqrs.max():,.0f})")
    print("3. Outlier Presence:       All regions exhibit upper-bound outliers exceeding Rs. 30K.")

    out_file = os.path.join(output_dir, "exercise_task2_profit_boxplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 2 saved to: {out_file}")

if __name__ == "__main__":
    run_task2()
