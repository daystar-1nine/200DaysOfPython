"""
Day 63 - Exercise 7: Exploratory Pairwise Matrix with Seaborn Pairplot
=====================================================================
Demonstrates full all-vs-all metric exploration using figure-level pairplot,
corner masking, and category hue partitioning.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task7():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="ticks")

    subset_df = df[["Quantity", "Discount", "Revenue", "Profit", "Category"]]

    # What is used: sns.pairplot with corner=True and hue="Category"
    # Why it is used: Rapidly scans pairwise scatter plots and diagonal univariate KDEs without redundant upper cells
    # How it works: Constructs a PairGrid object mapping variables across row and column axes
    g = sns.pairplot(
        subset_df,
        hue="Category",
        corner=True,
        diag_kind="kde",
        palette="tab10",
        plot_kws=dict(alpha=0.6, s=25, edgecolor="none")
    )

    g.fig.suptitle("E-Commerce Pairwise Metrics Matrix by Category", y=1.02, fontsize=14, fontweight="bold")

    out_file = os.path.join(output_dir, "exercise_task7_pairplot.png")
    g.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[SUCCESS] Task 7 saved to: {out_file}")

if __name__ == "__main__":
    run_task7()
