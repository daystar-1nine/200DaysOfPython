"""
Day 64 - Exercise 5: Categorical Faceting with Seaborn catplot
=============================================================
Demonstrates figure-level small multiples faceting using catplot(col="Category")
to evaluate whether regional performance remains consistent across product types.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_task5():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    # What is used: sns.catplot with col="Category" and kind="bar"
    # Why it is used: Creates synchronized small multiples across all categories
    # How it works: Instantiates FacetGrid and plots separate bar charts for each product category
    g = sns.catplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Region",
        legend=False,
        col="Category",
        col_wrap=3,
        kind="bar",
        palette="deep",
        height=3.8,
        aspect=1.1,
        errorbar=None
    )

    g.set_axis_labels("Region", "Mean Revenue (₹)")
    g.fig.suptitle("Task 5: Regional Mean Revenue Faceted Across Categories", y=1.02, fontsize=14, fontweight="bold")

    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    out_file = os.path.join(output_dir, "exercise_task5_faceted_catplot.png")
    g.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    print("[FINDING] Regional rankings shift by category: Electronics is led by North, while West leads Fitness and Apparel.")
    print(f"[SUCCESS] Task 5 saved to: {out_file}")

if __name__ == "__main__":
    run_task5()
