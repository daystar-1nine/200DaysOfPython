"""
Tests for Executive Report Authoring
====================================
"""

import os
from app.report import generate_advanced_eda_report

def test_generate_advanced_eda_report(clean_sample_df, tmp_path):
    out = str(tmp_path / "test_report.txt")
    text = generate_advanced_eda_report(clean_sample_df, out)
    assert os.path.exists(out)
    assert "SECTION 1 — DATASET OVERVIEW" in text
    assert "SECTION 9 — 15 STRATEGIC BUSINESS INSIGHTS" in text
    assert "Insight #15" in text
    assert len(text) > 3000
