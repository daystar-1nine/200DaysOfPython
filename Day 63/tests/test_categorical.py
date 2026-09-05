"""
Tests for Categorical Visualization Functions
============================================
"""

import os
from app.categorical import (
    plot_orders_by_category_and_segment,
    plot_regional_revenue_barplot
)

def test_plot_orders_by_category_and_segment(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_countplot.png")
    plot_orders_by_category_and_segment(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_regional_revenue_barplot(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_barplot.png")
    plot_regional_revenue_barplot(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
