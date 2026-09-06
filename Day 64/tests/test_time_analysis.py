"""
Tests for Longitudinal Time-Series Chart Renderers
==================================================
"""

import os
from app.time_analysis import (
    plot_monthly_revenue,
    plot_monthly_profit
)

def test_plot_monthly_revenue(clean_sample_df, tmp_path):
    p = str(tmp_path / "m_rev.png")
    plot_monthly_revenue(clean_sample_df, p)
    assert os.path.exists(p)

def test_plot_monthly_profit(clean_sample_df, tmp_path):
    p = str(tmp_path / "m_prof.png")
    plot_monthly_profit(clean_sample_df, p)
    assert os.path.exists(p)
