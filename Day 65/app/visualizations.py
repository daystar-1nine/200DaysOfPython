"""
Statistical data visualization module.
Generates publication-quality diagnostic figures comparing computed statistics with charts.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def generate_distribution_charts(df: pd.DataFrame, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    saved_charts = []
    
    # 1. Revenue Distribution (Histogram + KDE)
    rev_path = os.path.join(output_dir, "revenue_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rev_clean = df["Revenue"].dropna()
    mean_rev = rev_clean.mean()
    median_rev = rev_clean.median()
    sns.histplot(rev_clean, bins=35, kde=True, color="#2b5c8f", ax=ax, edgecolor="black", alpha=0.6)
    ax.axvline(mean_rev, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: Rs. {mean_rev:,.0f}")
    ax.axvline(median_rev, color="#27ae60", linestyle="-", linewidth=2, label=f"Median: Rs. {median_rev:,.0f}")
    ax.set_title("Revenue Distribution: Histogram & Gaussian KDE Overlay", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Revenue (INR)", fontsize=10)
    ax.set_ylabel("Order Count", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(rev_path, dpi=300)
    plt.close(fig)
    saved_charts.append(rev_path)
    
    # 2. Profit Distribution (Histogram + KDE)
    profit_path = os.path.join(output_dir, "profit_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    profit_clean = df["Profit"].dropna()
    mean_prof = profit_clean.mean()
    median_prof = profit_clean.median()
    sns.histplot(profit_clean, bins=35, kde=True, color="#16a085", ax=ax, edgecolor="black", alpha=0.6)
    ax.axvline(mean_prof, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: Rs. {mean_prof:,.0f}")
    ax.axvline(median_prof, color="#2980b9", linestyle="-", linewidth=2, label=f"Median: Rs. {median_prof:,.0f}")
    ax.set_title("Profit Distribution: Diagnostic Central Tendency & Tails", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Profit (INR)", fontsize=10)
    ax.set_ylabel("Order Count", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(profit_path, dpi=300)
    plt.close(fig)
    saved_charts.append(profit_path)
    
    # 3. Quantity Distribution (Discrete Frequency Count)
    qty_path = os.path.join(output_dir, "quantity_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    qty_clean = df["Quantity"].dropna()
    sns.countplot(x=qty_clean, color="#8e44ad", ax=ax, edgecolor="black", alpha=0.8)
    ax.set_title("Order Quantity Distribution: Discrete Item Frequency", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Quantity per Order", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig(qty_path, dpi=300)
    plt.close(fig)
    saved_charts.append(qty_path)
    
    # 4. Revenue Boxplot (IQR Outlier Fences)
    rev_box_path = os.path.join(output_dir, "revenue_boxplot.png")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    sns.boxplot(x=rev_clean, color="#3498db", flierprops={"marker": "D", "markerfacecolor": "#e74c3c", "markersize": 6}, ax=ax)
    q1 = rev_clean.quantile(0.25)
    q3 = rev_clean.quantile(0.75)
    iqr = q3 - q1
    uf = q3 + 1.5 * iqr
    ax.axvline(uf, color="#f39c12", linestyle=":", linewidth=2, label=f"Tukey Upper Fence (Rs. {uf:,.0f})")
    ax.set_title("Revenue Dispersion: Interquartile Range & Outlier Fliers", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Revenue (INR)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(rev_box_path, dpi=300)
    plt.close(fig)
    saved_charts.append(rev_box_path)
    
    # 5. Profit Boxplot
    prof_box_path = os.path.join(output_dir, "profit_boxplot.png")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    sns.boxplot(x=profit_clean, color="#2ecc71", flierprops={"marker": "D", "markerfacecolor": "#e74c3c", "markersize": 6}, ax=ax)
    pq1 = profit_clean.quantile(0.25)
    pq3 = profit_clean.quantile(0.75)
    piqr = pq3 - pq1
    puf = pq3 + 1.5 * piqr
    ax.axvline(puf, color="#d35400", linestyle=":", linewidth=2, label=f"Upper Fence (Rs. {puf:,.0f})")
    ax.set_title("Profit Dispersion & Outlier Envelope", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Profit (INR)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(prof_box_path, dpi=300)
    plt.close(fig)
    saved_charts.append(prof_box_path)
    
    # 6. Discount Boxplot
    disc_box_path = os.path.join(output_dir, "discount_boxplot.png")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    disc_clean = df["Discount"].dropna()
    sns.boxplot(x=disc_clean, color="#f1c40f", ax=ax)
    ax.set_title("Discount Policy Distribution Across Transactions", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Discount Rate (Fractional)", fontsize=10)
    plt.tight_layout()
    plt.savefig(disc_box_path, dpi=300)
    plt.close(fig)
    saved_charts.append(disc_box_path)
    
    return saved_charts
