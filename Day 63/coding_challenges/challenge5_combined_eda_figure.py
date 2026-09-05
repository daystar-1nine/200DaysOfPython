"""
Day 63 - Coding Challenge 5: Comprehensive 6-Panel Statistical EDA Figure
=======================================================================
Combines univariate distributions, categorical boxplots, aggregated bar plots,
frequency countplots, bivariate regressions, and masked correlation heatmaps
into a single publication-grade multi-panel master figure.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_challenge5():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid", palette="deep")

    # Allocate 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Revenue Distribution (Hist + KDE)
    sns.histplot(df["Revenue"], kde=True, bins=25, color="#1f77b4", ax=axes[0, 0])
    axes[0, 0].axvline(df["Revenue"].median(), color="crimson", linestyle="--", label=f"Med: ₹{df['Revenue'].median()*1e-3:.0f}K")
    axes[0, 0].set_title("1. Order Revenue Distribution (KDE)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Revenue (₹)")
    axes[0, 0].xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    axes[0, 0].legend()

    # 2. Profit Margin by Category (Boxplot)
    sns.boxplot(data=df, x="Category", y="Profit_Margin", palette="Set2", ax=axes[0, 1])
    axes[0, 1].set_title("2. Profit Margin % by Category (IQR)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Category")
    axes[0, 1].set_ylabel("Profit Margin")
    axes[0, 1].yaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # 3. Regional Revenue Mean (Barplot with 95% CI)
    sns.barplot(data=df, x="Region", y="Revenue", estimator="mean", errorbar=("ci", 95), palette="Blues_d", ax=axes[0, 2])
    axes[0, 2].set_title("3. Mean Revenue by Region (95% CI)", fontsize=11, fontweight="bold")
    axes[0, 2].set_ylabel("Mean Revenue (₹)")
    axes[0, 2].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # 4. Order Count by Category & Region (Countplot)
    sns.countplot(data=df, x="Category", hue="Region", palette="tab10", ax=axes[1, 0])
    axes[1, 0].set_title("4. Order Volume by Category & Region", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Category")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].legend(title="Region", fontsize=8)

    # 5. Revenue vs Profit Regression (Regplot)
    sns.regplot(data=df, x="Revenue", y="Profit", scatter_kws={"alpha": 0.4, "color": "#2ca02c"}, line_kws={"color": "darkred", "linewidth": 2}, ax=axes[1, 1])
    axes[1, 1].set_title("5. Revenue vs Profit (Linear Trend)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Revenue (₹)")
    axes[1, 1].set_ylabel("Profit (₹)")
    axes[1, 1].xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    axes[1, 1].yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    # 6. Masked Correlation Heatmap
    num_cols = ["Quantity", "Discount", "Revenue", "Cost", "Profit", "Profit_Margin"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=axes[1, 2])
    axes[1, 2].set_title("6. Financial Metrics Correlation Matrix", fontsize=11, fontweight="bold")

    fig.suptitle("Comprehensive Multi-Panel Statistical EDA Master Figure", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "challenge5_combined_eda.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 5 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge5()
