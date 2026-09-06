"""
Tests for Relational & Faceted Chart Renderers
=============================================
"""

import os
from app.relationships import (
    plot_region_category_revenue,
    plot_faceted_category_analysis,
    plot_revenue_profit,
    plot_category_relationships
)

def test_plot_region_category_revenue(clean_sample_df, tmp_path):
    p = str(tmp_path / "cat_hue.png")
    plot_region_category_revenue(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_faceted_category_analysis(clean_sample_df, tmp_path):
    p = str(tmp_path / "cat_facet.png")
    plot_faceted_category_analysis(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_revenue_profit(clean_sample_df, tmp_path):
    p = str(tmp_path / "rel_scatter.png")
    plot_revenue_profit(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_category_relationships(clean_sample_df, tmp_path):
    p = str(tmp_path / "rel_facet.png")
    plot_category_relationships(clean_sample_df, p)
    assert os.path.exists(p)
