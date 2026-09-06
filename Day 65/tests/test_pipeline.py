"""Tests for end-to-end report generation pipeline."""
import os
from app.report import generate_all_reports

def test_generate_all_reports(real_sales_df, tmp_path):
    target_cols = ["Quantity", "Revenue", "Profit"]
    res = generate_all_reports(real_sales_df, target_cols, str(tmp_path))
    
    assert os.path.exists(res["report_txt"])
    assert os.path.exists(os.path.join(str(tmp_path), "statistical_summary.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "percentile_report.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "outlier_report.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "zscore_report.csv"))
    
    assert len(res["basic_df"]) == 3
    assert len(res["insights"]) > 0
