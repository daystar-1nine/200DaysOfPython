"""
Correlation Matrix Heatmap Module
=================================
Visualizes lower-triangle Pearson correlation matrix with diverging colormap.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.config import DEFAULT_DPI

def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: str):
    """
    Chart 16: Correlation heatmap from precomputed correlation DataFrame.
    """
    sns.set_theme(style="white")
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7.5))

    sns.heatmap(
        corr_df,
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

    ax.set_title("Chart 16: E-Commerce Financial Metrics Correlation Matrix", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
