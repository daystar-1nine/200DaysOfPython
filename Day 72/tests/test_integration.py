"""
Integration tests for the complete application workflow.
"""
import os
import pytest
import pandas as pd
from app.config import AppConfig
from app.loader import load_dataset
from app.cleaner import clean_dataset
from app.validator import validate_numeric_data
from app.covariance import compute_covariance_matrix
from app.correlation_matrix import compute_correlation_matrices, compute_all_pairwise_results
from app.analyzer import RelationshipAnalyzer
from app.insights import generate_business_insights
from app.visualizations import generate_all_charts
from app.report import export_csv_results, format_ascii_report

def test_full_pipeline_execution(tmp_path):
    # Use real e-commerce data
    csv_path = os.path.join(r"s:\Programming\Python200days\Day 72\data", "ecommerce_sales.csv")
    assert os.path.exists(csv_path)
    
    raw = load_dataset(csv_path)
    cleaned = clean_dataset(raw)
    val = validate_numeric_data(cleaned)
    assert val["valid"] is True
    
    cov_mat = compute_covariance_matrix(cleaned)
    p_mat, s_mat, d_mat = compute_correlation_matrices(cleaned)
    pairwise_df = compute_all_pairwise_results(cleaned)
    
    analyzer = RelationshipAnalyzer(pairwise_df, cleaned)
    top_pos = analyzer.get_top_positive(5)
    top_neg = analyzer.get_top_negative(5)
    mc_df = analyzer.get_multicollinear_pairs()
    div_df = analyzer.get_divergent_pairs()
    q_and_a = analyzer.answer_business_questions()
    
    insights = generate_business_insights(top_pos, top_neg, mc_df)
    
    # Test export to temporary directory
    out_dir = str(tmp_path)
    export_csv_results(cov_mat, p_mat, pairwise_df, out_dir)
    assert os.path.exists(os.path.join(out_dir, "covariance_matrix.csv"))
    assert os.path.exists(os.path.join(out_dir, "correlation_matrix.csv"))
    assert os.path.exists(os.path.join(out_dir, "correlation_results.csv"))
    
    report_text = format_ascii_report(
        n_rows=val["n_rows"],
        n_cols=val["n_cols"],
        top_pos=top_pos,
        top_neg=top_neg,
        mc_df=mc_df,
        divergent_df=div_df,
        insights=insights,
        q_and_a=q_and_a
    )
    assert "BUSINESS RELATIONSHIP & CORRELATION ANALYSIS REPORT" in report_text
    assert len(report_text) > 500
