"""
Relational & Multidimensional Visualizations Module
===================================================
Generates figure-level catplot and relplot charts with hue, size, and faceting.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.config import DEFAULT_DPI, SNS_THEME

def plot_region_category_revenue(df: pd.DataFrame, output_path: str):
    """
    Chart 10: Revenue by Region + Category (catplot with hue).
    """
    sns.set_theme(style=SNS_THEME)
    g = sns.catplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Category",
        kind="bar",
        palette="Set2",
        height=5,
        aspect=1.4,
        errorbar=None
    )
    g.fig.suptitle("Chart 10: Revenue by Region & Category (Hue Encoding)", y=1.02, fontsize=13, fontweight="bold")
    g.set_axis_labels("Region", "Mean Revenue (Rs.)")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(g.fig)

def plot_faceted_category_analysis(df: pd.DataFrame, output_path: str):
    """
    Chart 11: Revenue by Region faceted by Category (catplot col="Category").
    """
    sns.set_theme(style=SNS_THEME)
    g = sns.catplot(
        data=df,
        x="Region",
        y="Revenue",
        hue="Region",
        legend=False,
        col="Category",
        col_wrap=3,
        kind="bar",
        palette="deep",
        height=3.8,
        aspect=1.1,
        errorbar=None
    )
    g.fig.suptitle("Chart 11: Regional Revenue Faceted by Product Category", y=1.02, fontsize=13, fontweight="bold")
    g.set_axis_labels("Region", "Mean Revenue (Rs.)")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(g.fig)

def plot_revenue_profit(df: pd.DataFrame, output_path: str):
    """
    Chart 12: Revenue vs Profit (relplot with hue=Region, size=Quantity).
    """
    sns.set_theme(style=SNS_THEME)
    g = sns.relplot(
        data=df,
        x="Revenue",
        y="Profit",
        hue="Region",
        size="Quantity",
        sizes=(30, 250),
        alpha=0.75,
        palette="tab10",
        height=5.5,
        aspect=1.4
    )
    g.fig.suptitle("Chart 12: Revenue vs Profit Scalability (Hue: Region, Size: Quantity)", y=1.02, fontsize=13, fontweight="bold")
    g.ax.xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
    g.ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(g.fig)

def plot_category_relationships(df: pd.DataFrame, output_path: str):
    """
    Chart 13: Revenue vs Profit by Category (relplot col="Category").
    """
    sns.set_theme(style=SNS_THEME)
    g = sns.relplot(
        data=df,
        x="Revenue",
        y="Profit",
        hue="Region",
        col="Category",
        col_wrap=3,
        height=3.8,
        aspect=1.15,
        alpha=0.8,
        palette="deep"
    )
    g.fig.suptitle("Chart 13: Revenue vs Profit Relational Slopes Faceted by Category", y=1.02, fontsize=13, fontweight="bold")
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")
        ax.yaxis.set_major_formatter(lambda x, pos: f"Rs. {x*1e-3:.0f}K")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(g.fig)
