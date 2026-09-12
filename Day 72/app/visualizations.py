"""
Day 72 — Visualization Engine
Renders 8+ publication-quality diagnostic charts using headless matplotlib Agg backend.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_all_charts(raw_df: pd.DataFrame, pearson_mat: pd.DataFrame, pairwise_df: pd.DataFrame, output_dir: str):
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(pearson_mat, dtype=bool))
    sns.heatmap(pearson_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Pearson Correlation (r)"}, ax=ax)
    ax.set_title("E-Commerce Feature Correlation Matrix (Upper Triangle Masked)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 2. Revenue vs Profit Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=raw_df, x="Revenue", y="Profit", scatter_kws={"alpha": 0.4, "color": "#1f77b4"},
                line_kws={"color": "#d62728", "linewidth": 2}, ax=ax)
    ax.set_title("Revenue vs. Profit Association", fontsize=12, fontweight="bold")
    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("Profit ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "revenue_profit.png"), dpi=300)
    plt.close()
    
    # 3. Quantity vs Revenue Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=raw_df, x="Quantity", y="Revenue", scatter_kws={"alpha": 0.4, "color": "#2ca02c"},
                line_kws={"color": "#1f77b4", "linewidth": 2}, ax=ax)
    ax.set_title("Quantity vs. Revenue Association", fontsize=12, fontweight="bold")
    ax.set_xlabel("Order Quantity")
    ax.set_ylabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "quantity_revenue.png"), dpi=300)
    plt.close()
    
    # 4. Discount vs Profit Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=raw_df, x="Discount", y="Profit", scatter_kws={"alpha": 0.4, "color": "#ff7f0e"},
                line_kws={"color": "#d62728", "linewidth": 2}, ax=ax)
    ax.set_title("Discount vs. Profit Association (Negative Co-Movement)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Discount Rate (0.00 - 1.00)")
    ax.set_ylabel("Profit ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "discount_profit.png"), dpi=300)
    plt.close()
    
    # 5. Cost vs Revenue Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=raw_df, x="Cost", y="Revenue", scatter_kws={"alpha": 0.4, "color": "#9467bd"},
                line_kws={"color": "#2ca02c", "linewidth": 2}, ax=ax)
    ax.set_title("Inventory Cost vs. Revenue Association", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "cost_revenue.png"), dpi=300)
    plt.close()
    
    # 6. Pearson vs Spearman Comparison Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pairwise_df["Pearson_r"], pairwise_df["Spearman_rho"], color="#3366cc", s=60, edgecolors="black", alpha=0.7)
    lims = [-1.0, 1.0]
    ax.plot(lims, lims, color="#d62728", linestyle="--", linewidth=1.5, label="Parity Line (r = rho)")
    ax.set_title("Pearson r vs. Spearman rho Across All Feature Pairs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation (Linear)")
    ax.set_ylabel("Spearman Correlation (Monotonic)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "pearson_spearman.png"), dpi=300)
    plt.close()
    
    # 7. Strongest Positive Relationships Horizontal Bar
    top_pos = pairwise_df[pairwise_df["Pearson_r"] > 0].sort_values(by="Pearson_r", ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r['Variable_1']} - {r['Variable_2']}" for _, r in top_pos.iterrows()]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, top_pos["Pearson_r"], color="#2ca02c", edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson Correlation (r)")
    ax.set_title("Strongest Positive Associations", fontsize=12, fontweight="bold")
    for i, v in enumerate(top_pos["Pearson_r"]):
        ax.text(v + 0.02, i, f"+{v:.3f}", va="center", fontweight="bold")
    ax.set_xlim(0, 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "strongest_positive.png"), dpi=300)
    plt.close()
    
    # 8. Strongest Negative Relationships Horizontal Bar
    top_neg = pairwise_df[pairwise_df["Pearson_r"] < 0].sort_values(by="Pearson_r", ascending=True).head(5)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels_neg = [f"{r['Variable_1']} - {r['Variable_2']}" for _, r in top_neg.iterrows()]
    y_neg = np.arange(len(labels_neg))
    ax.barh(y_neg, top_neg["Pearson_r"], color="#d62728", edgecolor="black", alpha=0.8)
    ax.set_yticks(y_neg)
    ax.set_yticklabels(labels_neg)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson Correlation (r)")
    ax.set_title("Strongest Inverse (Negative) Associations", fontsize=12, fontweight="bold")
    for i, v in enumerate(top_neg["Pearson_r"]):
        ax.text(v - 0.08, i, f"{v:.3f}", va="center", fontweight="bold")
    ax.set_xlim(-1.15, 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "strongest_negative.png"), dpi=300)
    plt.close()
    
    # 9. Executive Dashboard (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Subplot 1: Distribution of correlations
    axes[0, 0].hist(pairwise_df["Pearson_r"], bins=15, color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Pearson Correlations", fontweight="bold")
    axes[0, 0].set_xlabel("Pearson r")
    axes[0, 0].set_ylabel("Pair Count")
    
    # Subplot 2: Heatmap subset
    key_vars = ["Quantity", "Unit_Price", "Discount", "Revenue", "Profit"]
    sub_mat = raw_df[key_vars].corr()
    sns.heatmap(sub_mat, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("Core KPI Correlation Matrix", fontweight="bold")
    
    # Subplot 3: Revenue vs Profit
    axes[1, 0].scatter(raw_df["Revenue"], raw_df["Profit"], alpha=0.3, color="#2ca02c")
    axes[1, 0].set_title("Revenue vs Profit Co-Movement", fontweight="bold")
    axes[1, 0].set_xlabel("Revenue ($)")
    axes[1, 0].set_ylabel("Profit ($)")
    
    # Subplot 4: Parity scatter
    axes[1, 1].scatter(pairwise_df["Pearson_r"], pairwise_df["Spearman_rho"], color="#ff7f0e", edgecolors="black", alpha=0.7)
    axes[1, 1].plot([-1, 1], [-1, 1], "r--")
    axes[1, 1].set_title("Linear vs Rank Correlation Alignment", fontweight="bold")
    axes[1, 1].set_xlabel("Pearson r")
    axes[1, 1].set_ylabel("Spearman rho")
    
    plt.suptitle("Executive Relationship & Correlation Dashboard", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "dashboard.png"), dpi=300)
    plt.close()
    
    print(f"Generated 9 publication charts in {charts_dir}")
