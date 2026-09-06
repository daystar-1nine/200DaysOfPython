"""
Tests for Customer Analytics Chart Renderers
============================================
"""

import os
from app.analyzer import get_top_customers
from app.customer_analysis import plot_top_customers

def test_plot_top_customers(clean_sample_df, tmp_path):
    top_rev, top_prof, _ = get_top_customers(clean_sample_df, 5)
    p = str(tmp_path / "top_cust.png")
    plot_top_customers(top_rev, top_prof, p)
    assert os.path.exists(p)
