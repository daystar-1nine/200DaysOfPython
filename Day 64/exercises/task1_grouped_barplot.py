"""
Day 64 - Exercise 1: Grouped Barplot (Average Revenue by Region & Category)
==========================================================================
Demonstrates grouped statistical aggregation using barplot with hue encoding
to identify high-performing regional product category combinations.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task1():
    # What is used: Relative path resolution and Pandas CSV ingestion
    # Why it is used: Ensures reproducible execution across environments
    # How it works: Locates ecommerce_sales.csv relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    # What is used: Matplotlib Figure and Axes allocation
    # Why it is used: Precise canvas dimension control
    # How it works: Creates an 11x6 inch canvas
    fig, ax = plt.subplots(figsize=(11, 6))

    # What is used: sns.barplot with hue="Category"
    # Why it is used: Breaks down regional mean revenue across product categories
    # How it works: Computes mean revenue and 95% bootstrapped confidence intervals for each Region x Category pair
    sns.barplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Category",
        palette="Set2",
        estimator="mean",
        errorbar=("ci", 95),
        edgecolor="black",
        linewidth=0.6,
        ax=ax
    )

    # What is used: Currency axis formatter
    # Why it is used: Formats values into human-readable thousands (₹K)
    # How it works: Scales tick values by 1e-3 and prepends rupee symbol
    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    ax.set_title("Task 1: Average Order Revenue by Region & Product Category", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sales Region", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Revenue (₹)", fontsize=11, fontweight="bold")
    ax.legend(title="Product Category", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    sns.despine(ax=ax, top=True, right=True)

    # What is used: Programmatic finding of strongest combination
    # Why it is used: Answers the core analytical question objectively
    # How it works: Groups by Region and Category, calculates mean, and finds the maximum
    combo_means = df.groupby(["Region", "Category"])["Revenue"].mean()
    strongest = combo_means.idxmax()
    strongest_val = combo_means.max()

    print(f"[FINDING] Strongest Region-Category: {strongest[0]} - {strongest[1]} (Mean: Rs. {strongest_val:,.0f})")

    out_file = os.path.join(output_dir, "exercise_task1_grouped_barplot.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Task 1 saved to: {out_file}")

if __name__ == "__main__":
    run_task1()
