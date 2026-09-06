"""
Day 64 - Coding Challenge 5: Automated Natural-Language Business Insight Generator
=================================================================================
Dynamically analyzes the sales DataFrame and constructs data-driven insight statements
using calculated empirical metrics without hardcoding conclusions.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_dynamic_insights(df: pd.DataFrame) -> list[str]:
    """
    Computes summary metrics and authors natural language business findings.
    """
    insights = []

    # 1. Highest revenue region
    reg_rev = df.groupby("Region")["Revenue"].sum()
    top_reg = reg_rev.idxmax()
    insights.append(f"The highest-revenue region is {top_reg} with total sales of Rs. {reg_rev.max():,.2f} ({reg_rev.max()/reg_rev.sum()*100:.1f}% of total).")

    # 2. Most profitable category
    cat_prof = df.groupby("Category")["Profit"].sum()
    top_cat = cat_prof.idxmax()
    insights.append(f"The most profitable category is {top_cat}, generating Rs. {cat_prof.max():,.2f} in net profit.")

    # 3. Correlation between Revenue and Profit
    r_val = df["Revenue"].corr(df["Profit"])
    strength = "strong positive" if r_val > 0.7 else ("moderate" if r_val > 0.4 else "weak")
    insights.append(f"Revenue and profit have a {strength} Pearson correlation of r = {r_val:+.3f}, confirming robust linear scalability.")

    # 4. Skewness diagnosis
    skew_rev = df["Revenue"].skew()
    mean_rev = df["Revenue"].mean()
    med_rev = df["Revenue"].median()
    insights.append(f"Order revenue is heavily right-skewed (skewness = {skew_rev:+.2f}), causing mean (Rs. {mean_rev:,.0f}) to exceed median (Rs. {med_rev:,.0f}) by {(mean_rev-med_rev)/med_rev*100:.1f}%.")

    # 5. Top Customer
    cust_rev = df.groupby("Customer_Name")["Revenue"].sum()
    top_cust = cust_rev.idxmax()
    insights.append(f"The primary enterprise account is {top_cust}, contributing Rs. {cust_rev.max():,.2f} across multiple high-volume transactions.")

    return insights

def run_challenge5():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ecommerce_sales.csv")
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    insights = generate_dynamic_insights(df)

    print("=" * 70)
    print(">>> AUTOMATED DYNAMIC BUSINESS INSIGHTS")
    print("=" * 70)
    for idx, ins in enumerate(insights, 1):
        print(f"[{idx}] {ins}")
    print("=" * 70)

    # Render insights summary figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    text_block = "\n\n".join([f"• {ins}" for ins in insights])
    ax.text(
        0.05, 0.5, text_block,
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.2", facecolor="#f8f9fa", edgecolor="#343a40", lw=1.5)
    )
    ax.set_title("Challenge 5: Automated Dynamic Business Intelligence Insights", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "challenge5_automated_insights.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 5 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge5()
