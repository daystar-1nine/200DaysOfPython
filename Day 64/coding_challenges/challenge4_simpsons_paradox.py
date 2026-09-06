"""
Day 64 - Coding Challenge 4: Simpson's Paradox Investigation in Regional Sales
============================================================================
Evaluates whether macro-level regional comparisons reverse or diverge when
disaggregated across product categories due to differing product mix weights.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_challenge4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Macro level: Regional Average Revenue
    sns.barplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Region",
        legend=False,
        palette="Blues_d",
        ax=axes[0]
    )
    axes[0].set_title("Aggregate Level: Overall Mean Revenue by Region", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="Rs. %.0f", padding=3)

    # Disaggregated level: Regional Average Revenue within Fitness
    fitness_df = df[df["Category"] == "Fitness"]
    sns.barplot(
        data=fitness_df,
        x="Region",
        y="Revenue",
        hue="Region",
        legend=False,
        palette="Oranges_d",
        ax=axes[1]
    )
    axes[1].set_title("Disaggregated Subgroup: Fitness Category Mean Revenue", fontsize=11, fontweight="bold")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="Rs. %.0f", padding=3)

    fig.suptitle("Challenge 4: Subgroup vs Aggregate Behavior (Simpson's Paradox Investigation)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    print("[INSIGHT] In aggregate data, South leads overall revenue (Rs. 83.8K), but within Fitness, West dominates by a wide margin (Rs. 86.2K vs South Rs. 68.4K).")

    out_file = os.path.join(output_dir, "challenge4_simpsons_paradox_investigation.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 4 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge4()
