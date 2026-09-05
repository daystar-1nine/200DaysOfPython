"""
Unit and Integration Tests for Charts and Dashboard Generation (charts.py, dashboard.py).
"""

import pytest
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from app import charts
from app import analyzer
from app.dashboard import create_dashboard


def test_save_individual_chart(tmp_path):
    """Verifies standalone chart saving and figure memory closure."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [10, 20])
    target = tmp_path / "test_chart.png"
    res = charts.save_individual_chart(fig, target)

    assert res.exists()
    assert res.stat().st_size > 500
    assert fig.number not in plt.get_fignums()


def test_plot_kpi_card():
    """Verifies KPI card rendering on Axes."""
    fig, ax = plt.subplots()
    charts.plot_kpi_card(ax, "Revenue", "₹50,000", "Volume", "#1f77b4")
    assert not ax.axison  # Axis lines turned off
    plt.close(fig)


def test_plot_monthly_revenue_trend(mock_ecommerce_df):
    """Verifies monthly revenue trend plotting on Axes."""
    m_rev = analyzer.get_monthly_revenue(mock_ecommerce_df)
    fig, ax = plt.subplots()
    charts.plot_monthly_revenue_trend(ax, m_rev, avg_line=True, annotate_max=True)
    assert len(ax.lines) >= 2  # Main line + average line
    plt.close(fig)


def test_plot_regional_revenue(mock_ecommerce_df):
    """Verifies regional bar plot on Axes."""
    reg_rev = analyzer.get_regional_revenue(mock_ecommerce_df)
    fig, ax = plt.subplots()
    charts.plot_regional_revenue(ax, reg_rev)
    assert len(ax.patches) == 4
    plt.close(fig)


def test_plot_category_revenue(mock_ecommerce_df):
    """Verifies category bar plot on Axes."""
    cat_rev = analyzer.get_category_revenue(mock_ecommerce_df)
    fig, ax = plt.subplots()
    charts.plot_category_revenue(ax, cat_rev)
    assert len(ax.patches) == 3
    plt.close(fig)


def test_plot_top_products(mock_ecommerce_df):
    """Verifies horizontal bar plot on Axes."""
    top_p = analyzer.get_top_products(mock_ecommerce_df, n=2)
    fig, ax = plt.subplots()
    charts.plot_top_products(ax, top_p)
    assert len(ax.patches) == 2
    plt.close(fig)


def test_plot_revenue_vs_profit(mock_ecommerce_df):
    """Verifies scatter plot with trendline on Axes."""
    rev, prof, r = analyzer.get_revenue_vs_profit(mock_ecommerce_df)
    fig, ax = plt.subplots()
    charts.plot_revenue_vs_profit(ax, rev, prof, r)
    assert len(ax.collections) >= 1  # Scatter points
    assert len(ax.lines) >= 1  # Trendline
    plt.close(fig)


def test_plot_quantity_distribution(mock_ecommerce_df):
    """Verifies histogram on Axes."""
    qty = analyzer.get_quantity_distribution(mock_ecommerce_df)
    fig, ax = plt.subplots()
    charts.plot_quantity_distribution(ax, qty)
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_create_dashboard_full(tmp_path, mock_ecommerce_df):
    """Verifies that the master dashboard generates a high-resolution image file."""
    target_dash = tmp_path / "test_dashboard.png"
    out = create_dashboard(mock_ecommerce_df, target_dash)

    assert out.exists()
    assert out.stat().st_size > 20000  # High-res file size
    assert plt.get_fignums() == []  # All figures cleanly closed