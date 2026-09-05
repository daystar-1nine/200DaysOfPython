"""
Day 63 - Coding Challenge 4: Correlation Extremes Identification & Threshold Heatmap
==================================================================================
Extracts and ranks extreme Pearson correlation coefficients (top positive & negative)
and visualizes the lower-triangle correlation matrix with annotated extremes.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_challenge4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="white")

    numeric_cols = ["Quantity", "Unit_Price", "Cost_Price", "Discount", "Revenue", "Cost", "Profit", "Profit_Margin"]
    corr = df[numeric_cols].corr()

    # What is used: Unstacking correlation matrix to extract distinct non-diagonal pairs
    # Why it is used: Systematically ranks pairs to discover extreme positive and negative correlations
    # How it works: Converts square matrix to series, drops identity diagonal and symmetric duplicates
    corr_pairs = corr.unstack()
    unique_pairs = []
    seen = set()

    for (col1, col2), r_val in corr_pairs.items():
        if col1 != col2 and (col2, col1) not in seen:
            seen.add((col1, col2))
            unique_pairs.append({"Feature_A": col1, "Feature_B": col2, "Pearson_R": r_val})

    df_pairs = pd.DataFrame(unique_pairs).sort_values("Pearson_R", ascending=False)
    top_pos = df_pairs.head(3)
    top_neg = df_pairs.tail(3)

    print("--- Top 3 Positive Correlations ---")
    for _, row in top_pos.iterrows():
        print(f"  {row['Feature_A']} <-> {row['Feature_B']}: r = {row['Pearson_R']:+.3f}")

    print("\n--- Top 3 Negative Correlations ---")
    for _, row in top_neg.iterrows():
        print(f"  {row['Feature_A']} <-> {row['Feature_B']}: r = {row['Pearson_R']:+.3f}")

    # Heatmap visualization
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax
    )

    ax.set_title("E-Commerce Correlation Extremes Analysis", fontsize=14, fontweight="bold", pad=15)

    out_file = os.path.join(output_dir, "challenge4_correlation_extremes.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 4 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge4()
