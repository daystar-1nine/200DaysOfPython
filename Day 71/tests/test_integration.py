"""
Integration tests for ReportGenerator, Visualizer, and full pipeline.
"""

import os
import pandas as pd
import pytest

try:
    from app.report import ReportGenerator
    from app.visualizations import Visualizer
    from app.metrics import compute_conversion_metrics, compute_financial_metrics, compute_guardrail_metrics
    from app.proportion_tests import two_proportion_ztest
    from app.business_impact import project_annual_business_impact
    from app.decision import evaluate_ab_test_decision
except (ImportError, ModuleNotFoundError):
    from report import ReportGenerator
    from visualizations import Visualizer
    from metrics import compute_conversion_metrics, compute_financial_metrics, compute_guardrail_metrics
    from proportion_tests import two_proportion_ztest
    from business_impact import project_annual_business_impact
    from decision import evaluate_ab_test_decision

def test_full_pipeline_reports(tmp_path, sample_experiment_df):
    ctrl = sample_experiment_df[sample_experiment_df["group"] == "Control"]
    trt = sample_experiment_df[sample_experiment_df["group"] == "Treatment"]
    
    conv = compute_conversion_metrics(ctrl, trt)
    fin = compute_financial_metrics(ctrl, trt)
    guard = compute_guardrail_metrics(ctrl, trt)
    
    stats_res = two_proportion_ztest(conv["conversions_control"], conv["n_control"],
                                     conv["conversions_treatment"], conv["n_treatment"])
    stats_res["ci_lower"] = -0.01
    stats_res["ci_upper"] = 0.05
    
    impact = project_annual_business_impact(conv["absolute_lift"], fin["aov_treatment"] or 50.0)
    decision = evaluate_ab_test_decision(stats_res["p_value"], 0.05, conv["relative_lift_pct"], 8.0,
                                         guard["bounce_delta"], 0.05, guard["refund_delta"], 0.015)
                                         
    rep = ReportGenerator(output_dir=str(tmp_path))
    p_sum = rep.export_experiment_summary(conv, fin, guard, path=os.path.join(tmp_path, "summary.csv"))
    p_stat = rep.export_statistical_results(stats_res, path=os.path.join(tmp_path, "stats.csv"))
    p_biz = rep.export_business_impact_report(impact, path=os.path.join(tmp_path, "biz.txt"))
    p_full = rep.export_full_text_report(conv, fin, stats_res, guard, decision, impact, path=os.path.join(tmp_path, "full.txt"))
    
    assert os.path.exists(p_sum)
    assert os.path.exists(p_stat)
    assert os.path.exists(p_biz)
    assert os.path.exists(p_full)

def test_visualizer_generation(tmp_path, sample_experiment_df):
    viz = Visualizer(output_dir=str(tmp_path))
    ctrl = sample_experiment_df[sample_experiment_df["group"] == "Control"]
    trt = sample_experiment_df[sample_experiment_df["group"] == "Treatment"]
    conv = compute_conversion_metrics(ctrl, trt)
    fin = compute_financial_metrics(ctrl, trt)
    guard = compute_guardrail_metrics(ctrl, trt)
    stats_res = two_proportion_ztest(conv["conversions_control"], conv["n_control"],
                                     conv["conversions_treatment"], conv["n_treatment"])
    stats_res["ci_lower"] = -0.01
    stats_res["ci_upper"] = 0.05
    impact = project_annual_business_impact(conv["absolute_lift"], 50.0)
    decision = evaluate_ab_test_decision(stats_res["p_value"], 0.05, conv["relative_lift_pct"], 8.0,
                                         guard["bounce_delta"], 0.05, guard["refund_delta"], 0.015)
                                         
    charts = viz.generate_all(sample_experiment_df, conv, stats_res, decision, impact)
    assert len(charts) == 8
    for c in charts:
        assert os.path.exists(c)
