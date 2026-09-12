"""
Integration tests for the complete Sales Prediction Regression Engine.
"""
import os
import pytest
import numpy as np
import pandas as pd
from app.config import AppConfig
from app.loader import load_csv
from app.cleaner import clean_sales_data
from app.validator import validate_data
from app.eda import compute_eda
from app.regression import SimpleLinearRegressor
from app.predictions import generate_scenario_predictions
from app.metrics import evaluate_predictions
from app.residuals import analyze_residuals
from app.report import format_ascii_report, export_artifacts

def test_full_pipeline_workflow(tmp_path):
    # Verify dataset presence
    raw_path = os.path.join(r"s:\Programming\Python200days\Day 73\data\raw", "advertising_sales.csv")
    assert os.path.exists(raw_path)
    
    df_raw = load_csv(raw_path)
    cleaned_df = clean_sales_data(df_raw)
    val = validate_data(cleaned_df)
    assert val["valid"] is True
    
    eda = compute_eda(cleaned_df)
    assert eda["pearson_r"] > 0.80
    
    X = cleaned_df["Advertising_Spend"].values
    y = cleaned_df["Sales"].values
    
    reg = SimpleLinearRegressor().fit(X, y)
    y_pred = reg.predict(X)
    
    metrics = evaluate_predictions(y, y_pred)
    assert metrics["r2"] > 0.80
    
    res_info = analyze_residuals(y, y_pred)
    assert np.isclose(res_info["mean_residual"], 0.0, atol=1e-10)
    
    scenarios = generate_scenario_predictions(reg, [50000, 100000], val["min_feature"], val["max_feature"])
    assert len(scenarios) == 2
    
    # Export artifacts to temporary directory
    out_dir = str(tmp_path)
    metrics_summary = pd.DataFrame([{"Split": "All", **metrics}])
    report_text = format_ascii_report(
        eda, reg.get_params(), reg.get_equation(), metrics, metrics, res_info, scenarios, ["Sample insight."]
    )
    export_artifacts(out_dir, metrics_summary, scenarios, report_text)
    
    assert os.path.exists(os.path.join(out_dir, "regression_results.csv"))
    assert os.path.exists(os.path.join(out_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(out_dir, "regression_report.txt"))

def test_run_main_exit_code():
    from app.main import run_pipeline
    code = run_pipeline()
    assert code == 0
