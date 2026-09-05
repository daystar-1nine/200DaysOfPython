"""
Correlation Matrix & Pairwise Grid Module
=========================================
Computes and visualizes masked lower-triangle correlation heatmaps
and exploratory pairwise multivariate matrices.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.config import DEFAULT_DPI, NUMERIC_FEATURES

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 6: Masked Lower-Triangle Correlation Heatmap.
    """
    sns.set_theme(style="white")
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[cols].corr()

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
        linewidths=0.75,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax
    )

    ax.set_title("Fig 6: E-Commerce Financial Metrics Correlation (Masked Lower Triangle)", fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_multivariate_pairplot(df: pd.DataFrame, output_path: str):
    """
    Generates Fig 7: Pairplot matrix of key financial variables by Customer Segment.
    """
    sns.set_theme(style="ticks")
    subset_cols = ["Quantity", "Discount_Percent", "Revenue", "Profit", "Customer_Segment"]
    subset_df = df[subset_cols]

    g = sns.pairplot(
        subset_df,
        hue="Customer_Segment",
        corner=True,
        diag_kind="kde",
        palette="deep",
        plot_kws=dict(alpha=0.65, s=25, edgecolor="none")
    )

    g.fig.suptitle("Fig 7: Pairwise Relational Matrix by Customer Segment", y=1.02, fontsize=14, fontweight="bold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(g.fig)
