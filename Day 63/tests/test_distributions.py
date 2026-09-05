"""
Tests for Distribution Visualization Functions
==============================================
"""

import os
from app.distributions import (
    plot_order_value_distribution,
    plot_revenue_by_category_boxplot,
    plot_profit_margin_violin,
    plot_segment_strip_swarm
)

def test_plot_order_value_distribution(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_hist.png")
    plot_order_value_distribution(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_revenue_by_category_boxplot(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_box.png")
    plot_revenue_by_category_boxplot(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_profit_margin_violin(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_violin.png")
    plot_profit_margin_violin(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_segment_strip_swarm(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_strip.png")
    plot_segment_strip_swarm(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
