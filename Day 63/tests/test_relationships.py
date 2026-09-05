"""
Tests for Relational Visualization Functions
============================================
"""

import os
from app.relationships import (
    plot_order_value_vs_units_scatter,
    plot_order_value_vs_profit_regplot,
    plot_monthly_revenue_trend
)

def test_plot_order_value_vs_units_scatter(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_scatter.png")
    plot_order_value_vs_units_scatter(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_order_value_vs_profit_regplot(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_regplot.png")
    plot_order_value_vs_profit_regplot(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_monthly_revenue_trend(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_lineplot.png")
    plot_monthly_revenue_trend(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
