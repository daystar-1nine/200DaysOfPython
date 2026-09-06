"""
Tests for Correlation Heatmap Renderer
======================================
"""

import os
from app.analyzer import get_correlation_matrix
from app.correlation import plot_correlation_heatmap

def test_plot_correlation_heatmap(clean_sample_df, tmp_path):
    corr = get_correlation_matrix(clean_sample_df)
    p = str(tmp_path / "corr.png")
    plot_correlation_heatmap(corr, p)
    assert os.path.exists(p)
