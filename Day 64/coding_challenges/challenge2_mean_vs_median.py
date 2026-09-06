"""
Day 64 - Coding Challenge 2: Mean vs Median Skewness Diagnostics
===============================================================
Isolates variables where Mean != Median, quantifies empirical tail skewness,
and explains why arithmetic averages mislead executive decision-makers.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_challenge2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    num_cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Profit"]
    divergences = []

    for c in num_cols:
        mean_v = df[c].mean()
        med_v = df[c].median()
        skew_v = df[c].skew()
        pct_diff = ((mean_v - med_v) / med_v) * 100 if med_v != 0 else 0
        divergences.append({
            "Feature": c,
            "Mean": mean_v,
            "Median": med_v,
            "Skewness": skew_v,
            "Divergence_Pct": pct_diff
        })

    div_df = pd.DataFrame(divergences).sort_values("Divergence_Pct", ascending=False)
    print("--- Numeric Features Mean vs Median Divergence ---")
    for _, row in div_df.iterrows():
        print(f"  {row['Feature']:<12}: Mean={row['Mean']:>10.2f} | Median={row['Median']:>10.2f} | Skew={row['Skewness']:>+5.2f} | Diff={row['Divergence_Pct']:>+6.1f}%")

    # Dual distribution visualization for Revenue
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(df["Revenue"], kde=True, color="#1f77b4", bins=30, ax=axes[0])
    mean_r = df["Revenue"].mean()
    med_r = df["Revenue"].median()
    axes[0].axvline(mean_r, color="crimson", linestyle="--", linewidth=2.0, label=f"Mean: Rs. {mean_r:,.0f}")
    axes[0].axvline(med_r, color="green", linestyle="-.", linewidth=2.0, label=f"Median: Rs. {med_r:,.0f}")
    axes[0].set_title("Revenue Histogram & Gaussian KDE (Right-Skewed)", fontsize=11, fontweight="bold")
    axes[0].xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    axes[0].legend()

    # Empirical CDF plot
    sns.ecdfplot(df["Revenue"], color="#ff7f0e", linewidth=2.2, ax=axes[1])
    axes[1].axvline(med_r, color="green", linestyle="-.", label="50th Percentile (Median)")
    axes[1].axvline(mean_r, color="crimson", linestyle="--", label="Mean Position (~74th Pct)")
    axes[1].set_title("Empirical Cumulative Distribution Function (ECDF)", fontsize=11, fontweight="bold")
    axes[1].xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    axes[1].legend()

    fig.suptitle("Challenge 2: Mean vs Median Skewness Diagnostics for Order Revenue", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "challenge2_mean_vs_median_skewness.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 2 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge2()
