"""
Tests for Categorical Chart Renderers
====================================
"""

import os
from app.analyzer import (
    get_regional_revenue_stats,
    get_category_profit_stats,
    get_region_counts
)
from app.categorical import (
    plot_revenue_region_boxplot,
    plot_profit_category_boxplot,
    plot_quantity_category_violin,
    plot_regional_revenue,
    plot_category_profit,
    plot_region_counts
)

def test_plot_revenue_region_boxplot(clean_sample_df, tmp_path):
    p = str(tmp_path / "box_reg.png")
    plot_revenue_region_boxplot(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_profit_category_boxplot(clean_sample_df, tmp_path):
    p = str(tmp_path / "box_cat.png")
    plot_profit_category_boxplot(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_quantity_category_violin(clean_sample_df, tmp_path):
    p = str(tmp_path / "violin_cat.png")
    plot_quantity_category_violin(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_regional_revenue(clean_sample_df, tmp_path):
    reg_data = get_regional_revenue_stats(clean_sample_df)
    p = str(tmp_path / "bar_reg.png")
    plot_regional_revenue(reg_data, p)
    assert os.path.exists(p)

def test_plot_category_profit(clean_sample_df, tmp_path):
    cat_data = get_category_profit_stats(clean_sample_df)
    p = str(tmp_path / "bar_cat.png")
    plot_category_profit(cat_data, p)
    assert os.path.exists(p)

def test_plot_region_counts(clean_sample_df, tmp_path):
    counts_data = get_region_counts(clean_sample_df)
    p = str(tmp_path / "count_reg.png")
    plot_region_counts(counts_data, p)
    assert os.path.exists(p)
