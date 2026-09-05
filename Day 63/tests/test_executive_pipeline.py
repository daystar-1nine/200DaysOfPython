"""
Tests for Master Pipeline, Dashboard & Executive Report
=======================================================
"""

import os
from app.charts import generate_executive_dashboard, generate_all_charts
from app.report import generate_statistical_report

def test_generate_executive_dashboard(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_dashboard.png")
    generate_executive_dashboard(clean_sample_df, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 50000

def test_generate_all_charts_count(clean_sample_df, tmp_path):
    charts = generate_all_charts(clean_sample_df, str(tmp_path))
    assert len(charts) == 12
    for p in charts:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 1000

def test_generate_statistical_report_content(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_report.txt")
    text = generate_statistical_report(clean_sample_df, out)
    assert os.path.exists(out)
    assert "SEABORN STATISTICAL DATA VISUALIZATION" in text
    assert "Q1:" in text
    assert "Q10:" in text
    assert len(text) > 1000
