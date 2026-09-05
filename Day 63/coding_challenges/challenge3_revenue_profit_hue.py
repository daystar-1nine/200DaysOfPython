"""
Day 63 - Coding Challenge 3: Multivariate Scatter with Hue, Size, Style & Trendline
=================================================================================
Visualizes 5-dimensional relationship: Revenue, Profit, Category, Quantity, Region,
along with an ordinary least squares regression trendline.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_challenge3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    # What is used: sns.scatterplot with hue, size, and style mappings
    # Why it is used: Encodes 5 dimensions into a single readable scatter representation
    # How it works: Maps (X, Y) to financial metrics, hue to Category, size to Quantity, and style to Region
    sns.scatterplot(
        data=df,
        x="Revenue",
        y="Profit",
        hue="Category",
        size="Quantity",
        style="Region",
        sizes=(40, 300),
        alpha=0.75,
        palette="tab10",
        ax=ax
    )

    # What is used: Linear regression trendline fit with NumPy polyfit
    # Why it is used: Quantifies the macroscopic profitability gradient across all transactions
    # How it works: Computes slope m and intercept b minimizing squared residuals
    slope, intercept = np.polyfit(df["Revenue"], df["Profit"], 1)
    corr = df["Revenue"].corr(df["Profit"])
    r_squared = corr ** 2

    x_vals = np.linspace(df["Revenue"].min(), df["Revenue"].max(), 100)
    y_vals = slope * x_vals + intercept

    ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=2.0, label=f"Trendline: y={slope:.2f}x + {intercept:,.0f} (R²={r_squared:.2f})")

    ax.xaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")
    ax.yaxis.set_major_formatter(lambda x, pos: f"₹{x*1e-3:.0f}K")

    ax.set_title("Multivariate Profitability Analysis (Hue: Category, Size: Quantity, Style: Region)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Order Revenue (₹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Profit (₹)", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    sns.despine(ax=ax, top=True, right=True)

    out_file = os.path.join(output_dir, "challenge3_revenue_profit_multivariate.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 3 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge3()
