"""
Tests for Correlation & Pairplot Visualization Functions
========================================================
"""

import os
from app.correlation import (
    plot_correlation_heatmap,
    plot_multivariate_pairplot
)

def test_plot_correlation_heatmap(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_heatmap.png")
    plot_correlation_heatmap(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

def test_plot_multivariate_pairplot(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_pairplot.png")
    plot_multivariate_pairplot(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
