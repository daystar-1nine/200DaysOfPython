"""
Tests for Distribution Chart Renderers
======================================
"""

import os
from app.distributions import (
    plot_revenue_distribution,
    plot_profit_distribution,
    plot_quantity_distribution
)

def test_plot_revenue_distribution(clean_sample_df, tmp_path):
    p = str(tmp_path / "rev_dist.png")
    plot_revenue_distribution(clean_sample_df, p)
    assert os.path.exists(p)
    assert os.path.getsize(p) > 1000

def test_plot_profit_distribution(clean_sample_df, tmp_path):
    p = str(tmp_path / "prof_dist.png")
    plot_profit_distribution(clean_sample_df, p)
    assert os.path.exists(p)
    assert os.path.getsize(p) > 1000

def test_plot_quantity_distribution(clean_sample_df, tmp_path):
    p = str(tmp_path / "qty_dist.png")
    plot_quantity_distribution(clean_sample_df, p)
    assert os.path.exists(p)
    assert os.path.getsize(p) > 1000
