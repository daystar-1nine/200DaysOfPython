"""
Day 64 - Coding Challenge 1: Faceted Revenue vs Profit Across Product Categories
==============================================================================
Creates a publication-grade small multiples grid analyzing the relationship
between Revenue and Profit, conditioned on Category and colored by Region.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_challenge1():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    # What is used: sns.relplot with col="Category" and hue="Region"
    # Why it is used: Analyzes whether the revenue-profit slope and financial ceiling vary by category
    # How it works: Generates a FacetGrid of scatterplots sharing synchronized axes
    g = sns.relplot(
        data=df,
        x="Revenue",
        y="Profit",
        hue="Region",
        col="Category",
        col_wrap=3,
        palette="tab10",
        height=3.8,
        aspect=1.15,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.7
    )

    g.fig.suptitle("Challenge 1: Revenue vs Profit Faceted by Category & Colored by Region", y=1.02, fontsize=13, fontweight="bold")
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
        ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")

    out_file = os.path.join(output_dir, "challenge1_faceted_revenue_profit.png")
    g.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[SUCCESS] Challenge 1 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge1()
