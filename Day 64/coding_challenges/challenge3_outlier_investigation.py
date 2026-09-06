"""
Day 64 - Coding Challenge 3: Root-Cause Investigation of Top 1% Revenue Outliers
=============================================================================
Identifies the top 1% highest revenue transactions (99th percentile) and isolates
their structural drivers across Category, Region, Product, and Quantity.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_challenge3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    p99_threshold = df["Revenue"].quantile(0.99)
    top_outliers = df[df["Revenue"] >= p99_threshold].sort_values("Revenue", ascending=False)

    print(f"[DIAGNOSTIC] 99th Percentile Revenue Threshold: Rs. {p99_threshold:,.2f}")
    print(f"[DIAGNOSTIC] Total Top 1% Outliers Found: {len(top_outliers)} orders\n")
    print("--- Top 1% Transactions Summary ---")
    for _, row in top_outliers.head(5).iterrows():
        print(f"  {row['Order_ID']}: {row['Category']} - {row['Product']} | Qty: {row['Quantity']} | Rev: Rs. {row['Revenue']:,.0f} | Profit: Rs. {row['Profit']:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Outlier Category breakdown
    sns.countplot(
        data=top_outliers,
        x="Category",
        hue="Category",
        legend=False,
        palette="Reds_r",
        edgecolor="black",
        ax=axes[0]
    )
    axes[0].set_title("Top 1% Outlier Volume by Product Category", fontsize=11, fontweight="bold")
    for c in axes[0].containers:
        axes[0].bar_label(c, fontsize=10, fontweight="bold")

    # Panel 2: Revenue vs Quantity of Outliers
    sns.scatterplot(
        data=top_outliers,
        x="Quantity",
        y="Revenue",
        hue="Category",
        size="Profit",
        sizes=(60, 300),
        palette="Set1",
        ax=axes[1]
    )
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    axes[1].set_title("Top 1% Outliers: Revenue vs Quantity & Profit", fontsize=11, fontweight="bold")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Challenge 3: Root-Cause Investigation of Top 1% Outlier Orders", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "challenge3_top_outliers_breakdown.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 3 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge3()
