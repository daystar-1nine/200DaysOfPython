"""
Integration tests for DataLoader, Scenarios, and Reporting.
"""

import os
import pandas as pd
import pytest

try:
    from app.config import config
    from app.loader import DataLoader
    from app.scenarios import run_all_scenarios
    from app.report import ReportGenerator
except (ImportError, ModuleNotFoundError):
    from config import config
    from loader import DataLoader
    from scenarios import run_all_scenarios
    from report import ReportGenerator

def test_data_loader_all_datasets():
    loader = DataLoader()
    df_del = loader.load_delivery_times()
    assert len(df_del) == 250
    assert "delivery_time_minutes" in df_del.columns

    df_conv = loader.load_conversion_data()
    assert len(df_conv) == 1000
    assert "converted" in df_conv.columns

    df_mfg = loader.load_manufacturing()
    assert len(df_mfg) == 300
    assert "weight_grams" in df_mfg.columns

    df_ord = loader.load_customer_orders()
    assert len(df_ord) == 500
    assert "order_total_usd" in df_ord.columns

def test_run_all_scenarios():
    results = run_all_scenarios()
    assert len(results) == 4
    for r in results:
        assert r.sample_size > 0
        assert not pd.isna(r.point_estimate)
        assert not pd.isna(r.test_statistic)
        assert 0.0 <= r.p_value <= 1.0

def test_report_generation(tmp_path):
    results = run_all_scenarios()
    rep = ReportGenerator(output_dir=str(tmp_path))
    csv_file = os.path.join(tmp_path, "test_out.csv")
    txt_file = os.path.join(tmp_path, "test_out.txt")

    rep.export_results_csv(results, output_path=csv_file)
    assert os.path.exists(csv_file)
    df = pd.read_csv(csv_file)
    assert len(df) == 4

    rep.export_text_report(results, output_path=txt_file)
    assert os.path.exists(txt_file)
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "EXECUTIVE SUMMARY REPORT" in content
