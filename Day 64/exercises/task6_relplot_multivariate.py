"""
Day 64 - Exercise 6: Multivariate Relational Analysis with relplot
==================================================================
Maps 4 dimensions simultaneously: Revenue (X), Profit (Y), Region (Hue),
and Quantity (Size) using Seaborn's figure-level relplot interface.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task6():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    # What is used: sns.relplot with hue and size encodings
    # Why it is used: Analyzes financial scalability across territories and order volumes
    # How it works: Allocates FacetGrid and renders scatter markers scaled by Quantity
    g = sns.relplot(
        data=df,
        x="Revenue",
        y="Profit",
        hue="Region",
        size="Quantity",
        sizes=(30, 260),
        alpha=0.75,
        palette="tab10",
        height=5.5,
        aspect=1.4
    )

    g.ax.xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    g.ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    g.fig.suptitle("Task 6: Revenue vs Profit (Hue: Region, Size: Quantity)", y=1.02, fontsize=14, fontweight="bold")

    print("--- Answers to Questions ---")
    print("1. Association: Revenue is strongly positively associated with profit (linear scalability).")
    print("2. High-Value Deals: South and North capture several extreme transactions (>Rs. 250K).")
    print("3. Quantity vs Profit: Large quantities do NOT always guarantee high profit if unit prices or margins are low.")

    out_file = os.path.join(output_dir, "exercise_task6_relplot_multivariate.png")
    g.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[SUCCESS] Task 6 saved to: {out_file}")

if __name__ == "__main__":
    run_task6()
