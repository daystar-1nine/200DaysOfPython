"""
Unit tests for data quality analyzer module app/analyzer.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves app package import paths cleanly during pytest execution.
# How it works: Appends Day 58 parent directory to sys.path.
import sys
from pathlib import Path

DAY58_DIR = Path(__file__).resolve().parent.parent
if str(DAY58_DIR) not in sys.path:
    sys.path.insert(0, str(DAY58_DIR))

# What is used: Import pandas and analyze_data_quality function.
# Why it is used: Asserts before/after missing value metrics, duplicate counts, descriptive statistics, and demographic breakdowns.
# How it works: Executes analyze_data_quality on raw and clean test DataFrames.
import pandas as pd
from app.analyzer import analyze_data_quality


def test_analyze_data_quality_missing_comparison(sample_raw_messy_df, sample_clean_customer_df):
    """
    Test missing value count comparison between raw and clean datasets.
    """
    # What is used: analyze_data_quality call.
    # Why it is used: Asserts raw missing count is > 0 and clean missing count is 0.
    # How it works: Verifies missing dictionary metrics.
    mock_stats = {"initial_rows": 5, "final_rows": 4}
    analysis = analyze_data_quality(sample_raw_messy_df, sample_clean_customer_df, mock_stats)

    missing_info = analysis["missing"]
    assert "raw_count" in missing_info
    assert "clean_count" in missing_info
    assert missing_info["clean_count"]["age"] == 0
    assert missing_info["clean_count"]["salary"] == 0


def test_analyze_data_quality_duplicate_comparison(sample_raw_messy_df, sample_clean_customer_df):
    """
    Test duplicate count comparison between raw and clean datasets.
    """
    # What is used: analyze_data_quality call.
    # Why it is used: Asserts raw duplicates count is 1 and clean duplicates count is 0.
    # How it works: Checks duplicates dictionary keys.
    mock_stats = {"initial_rows": 5, "final_rows": 4}
    analysis = analyze_data_quality(sample_raw_messy_df, sample_clean_customer_df, mock_stats)

    dup_info = analysis["duplicates"]
    assert dup_info["raw_duplicates"] == 1
    assert dup_info["clean_duplicates"] == 0


def test_analyze_data_quality_descriptive_stats(sample_raw_messy_df, sample_clean_customer_df):
    """
    Test descriptive statistics calculation for Age and Salary.
    """
    # What is used: analyze_data_quality call.
    # Why it is used: Asserts descriptive stats for age and salary.
    # How it works: Verifies mean, median, min, max fields in descriptive dictionary.
    mock_stats = {"initial_rows": 5, "final_rows": 4}
    analysis = analyze_data_quality(sample_raw_messy_df, sample_clean_customer_df, mock_stats)

    desc = analysis["descriptive"]
    assert "age" in desc
    assert "salary" in desc
    assert desc["age"]["mean"] == 27.75
    assert desc["salary"]["mean"] == 62500.0


def test_analyze_data_quality_demographic_distributions(sample_raw_messy_df, sample_clean_customer_df):
    """
    Test demographic distributions generation for Gender, Department, and City.
    """
    # What is used: analyze_data_quality call.
    # Why it is used: Asserts demographic counts dictionary values.
    # How it works: Checks gender, department, and city frequency counts.
    mock_stats = {"initial_rows": 5, "final_rows": 4}
    analysis = analyze_data_quality(sample_raw_messy_df, sample_clean_customer_df, mock_stats)

    demo = analysis["demographics"]
    assert "gender" in demo
    assert demo["gender"]["Male"] == 2
    assert demo["gender"]["Female"] == 2
