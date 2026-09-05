"""
Day 63 - Coding Challenge 2: Department Salary Distributions (Boxplot vs Violin Plot)
===================================================================================
Side-by-side architectural comparison of Boxplot (Tukey 5-number summary & outliers)
versus Violin Plot (density shape, bimodal clusters, inner quartiles).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_challenge2():
    np.random.seed(101)
    depts = ["Engineering", "Product", "Sales", "Marketing", "Operations", "HR"]
    records = []

    for dept in depts:
        if dept == "Engineering":
            # Bimodal salary structure (Junior engineers vs Principal staff)
            salaries = np.concatenate([
                np.random.normal(loc=750000, scale=80000, size=60),
                np.random.normal(loc=1650000, scale=120000, size=40)
            ])
        elif dept == "Sales":
            # Right skewed with commission outliers
            salaries = np.random.exponential(scale=200000, size=100) + 500000
        else:
            base = 600000 if dept in ["HR", "Operations"] else 900000
            salaries = np.random.normal(loc=base, scale=100000, size=100)

        for s in salaries:
            records.append({"Department": dept, "Salary": max(300000, s)})

    df_dept = pd.DataFrame(records)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="Set2")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # What is used: sns.boxplot on axes[0]
    # Why it is used: Visualizes medians, IQRs, and isolates high-compensation outliers
    # How it works: Computes quartiles per department with whisker threshold at 1.5*IQR
    sns.boxplot(
        data=df_dept,
        x="Department",
        y="Salary",
        palette="Set2",
        ax=axes[0],
        flierprops=dict(marker="o", markersize=5, markerfacecolor="red", markeredgecolor="darkred")
    )
    axes[0].set_title("Department Salaries (Boxplot: Quartiles & Outliers)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Department", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Annual Compensation (₹)", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-5:.1f}L")
    axes[0].tick_params(axis="x", rotation=25)

    # What is used: sns.violinplot on axes[1]
    # Why it is used: Reveals bimodal density structure (e.g. Engineering tiering) hidden by boxplots
    # How it works: Overlays mirrored Gaussian KDE with inner quartile dashed lines
    sns.violinplot(
        data=df_dept,
        x="Department",
        y="Salary",
        palette="Set2",
        inner="quartile",
        cut=0,
        ax=axes[1]
    )
    axes[1].set_title("Department Salaries (Violin: Density & Multimodality)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Department", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Annual Compensation (₹)", fontsize=11, fontweight="bold")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-5:.1f}L")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("Compensation Structure Diagnostic: Boxplot vs Violin Plot", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "challenge2_dept_salaries.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 2 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge2()
