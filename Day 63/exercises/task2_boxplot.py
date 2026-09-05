"""
Day 63 - Exercise 2: Outlier Detection & Quartile Anatomy with Boxplot
====================================================================
Demonstrates the Tukey 5-number summary across product categories,
custom flier styling, and IQR outlier boundary calculations.
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

    fig, ax = plt.subplots(figsize=(11, 6))

    # What is used: sns.boxplot with custom flierprops
    # Why it is used: Visualizes median, IQR, and isolates anomalous outlier transactions
    # How it works: Computes quartiles per category and draws whiskers at 1.5 * IQR
    sns.boxplot(
        data=df,
        x="Category",
        y="Revenue",
        palette="Set2",
        ax=ax,
        flierprops=dict(marker="D", markersize=6, markerfacecolor="red", markeredgecolor="darkred", alpha=0.7),
        boxprops=dict(alpha=0.85),
        medianprops=dict(color="black", linewidth=2.0)
    )

    # What is used: Formatting y-axis currency with Lambda/FuncFormatter
    # Why it is used: Converts raw values to readable currency notation
    # How it works: Formats tick values to thousands (₹K)
    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    ax.set_title("Order Revenue by Product Category (IQR Outlier Analysis)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    sns.despine(ax=ax, top=True, right=True)

    out_file = os.path.join(output_dir, "exercise_task2_boxplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 2 saved to: {out_file}")

if __name__ == "__main__":
    run_task2()
