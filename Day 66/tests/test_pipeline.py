"""
End-to-end integration tests for Day 66 analytical pipeline.
"""
from pathlib import Path
import pytest
from app.analysis import run_full_pipeline
from app.report import generate_ascii_report
from app.config import OUTPUT_DIR, CHARTS_DIR

def test_run_full_pipeline_structure():
    res = run_full_pipeline()
    assert "coin" in res
    assert "dice" in res
    assert "dice_two" in res
    assert "cards" in res
    assert "conversion" in res
    assert "risk" in res
    assert "fraud_bayes" in res

def test_ascii_report_generation(tmp_path):
    res = run_full_pipeline()
    test_report_path = tmp_path / "test_report.txt"
    report_text = generate_ascii_report(res, output_path=test_report_path)
    
    assert test_report_path.exists()
    assert "DAY 66: REAL-WORLD PROBABILITY SIMULATOR EXECUTIVE REPORT" in report_text
    assert "LAW OF LARGE NUMBERS" in report_text
    assert "END OF REPORT" in report_text

def test_generated_output_files_exist():
    # Verify core artifacts exist in Day 66 output/
    assert (OUTPUT_DIR / "coin_results.csv").exists()
    assert (OUTPUT_DIR / "dice_results.csv").exists()
    assert (OUTPUT_DIR / "card_results.csv").exists()
    assert (OUTPUT_DIR / "conversion_results.csv").exists()
    assert (OUTPUT_DIR / "probability_report.txt").exists()
    
    # Verify charts
    assert (CHARTS_DIR / "coin_convergence.png").exists()
    assert (CHARTS_DIR / "dice_distribution.png").exists()
    assert (CHARTS_DIR / "dice_sum_distribution.png").exists()
    assert (CHARTS_DIR / "expected_vs_actual.png").exists()
    assert (CHARTS_DIR / "fraud_probability.png").exists()
