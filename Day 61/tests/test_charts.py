"""
Unit Tests for Chart Generation and Visualization Engine (charts.py).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from app import charts
from app import analyzer


def test_save_chart_utility(tmp_path):
    """Tests that save_chart exports file, adjusts layout, and closes figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    target = tmp_path / "test_save.png"
    result = charts.save_chart(fig, target)

    assert result.exists()
    assert result.stat().st_size > 0
    # Figure should be closed
    assert fig.number not in plt.get_fignums()


def test_plot_monthly_revenue(tmp_path, sample_sales_df):
    """Tests Chart 1 monthly revenue line chart."""
    data = analyzer.get_monthly_revenue(sample_sales_df)
    target = tmp_path / "c1_monthly.png"
    res = charts.plot_monthly_revenue(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_revenue_by_region(tmp_path, sample_sales_df):
    """Tests Chart 2 regional revenue bar chart."""
    data = analyzer.get_revenue_by_region(sample_sales_df)
    target = tmp_path / "c2_region.png"
    res = charts.plot_revenue_by_region(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_revenue_by_category(tmp_path, sample_sales_df):
    """Tests Chart 3 category revenue bar chart."""
    data = analyzer.get_revenue_by_category(sample_sales_df)
    target = tmp_path / "c3_category.png"
    res = charts.plot_revenue_by_category(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_top_products(tmp_path, sample_sales_df):
    """Tests Chart 4 top products horizontal bar chart."""
    data = analyzer.get_top_n_products(sample_sales_df, n=3)
    target = tmp_path / "c4_products.png"
    res = charts.plot_top_products(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_top_customers(tmp_path, sample_sales_df):
    """Tests Chart 5 top customers horizontal bar chart."""
    data = analyzer.get_top_n_customers(sample_sales_df, n=3)
    target = tmp_path / "c5_customers.png"
    res = charts.plot_top_customers(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_quantity_distribution(tmp_path, sample_sales_df):
    """Tests Chart 6 quantity distribution histogram."""
    data = analyzer.get_quantity_distribution(sample_sales_df)
    target = tmp_path / "c6_qty.png"
    res = charts.plot_quantity_distribution(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_revenue_vs_profit(tmp_path, sample_sales_df):
    """Tests Chart 7 revenue vs profit scatter plot."""
    rev, prof, r = analyzer.get_revenue_vs_profit(sample_sales_df)
    target = tmp_path / "c7_scatter.png"
    res = charts.plot_revenue_vs_profit(rev, prof, r, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_category_share(tmp_path, sample_sales_df):
    """Tests Chart 8 category share pie chart."""
    data = analyzer.get_category_revenue_share(sample_sales_df)
    target = tmp_path / "c8_pie.png"
    res = charts.plot_category_share(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_plot_monthly_rolling(tmp_path, sample_sales_df):
    """Tests Chart 9 bonus monthly rolling line chart."""
    data = analyzer.get_monthly_rolling_revenue(sample_sales_df, window=3)
    target = tmp_path / "c9_rolling.png"
    res = charts.plot_monthly_rolling(data, target)
    assert res.exists()
    assert res.stat().st_size > 1000
