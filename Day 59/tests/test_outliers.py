"""
Unit Tests for app/outliers.py and app/descriptive.py modules.
"""

from app.cleaner import clean_sales_data
from app.descriptive import compute_descriptive_stats
from app.outliers import detect_all_outliers
from app.transformer import compute_derived_metrics


def test_compute_descriptive_stats(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    desc = compute_descriptive_stats(enriched_df)

    assert "Revenue" in desc
    assert "mean" in desc["Revenue"]
    assert "q1_25%" in desc["Revenue"]
    assert "iqr" in desc["Revenue"]


def test_detect_all_outliers(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    outliers = detect_all_outliers(enriched_df)

    assert "Revenue" in outliers
    assert "lower_bound" in outliers["Revenue"]
    assert "upper_bound" in outliers["Revenue"]
    assert "outliers" in outliers["Revenue"]


def test_detect_all_outliers_iqr_bounds(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    outliers = detect_all_outliers(enriched_df)

    for col, res in outliers.items():
        assert res["q3"] >= res["q1"]
        assert res["iqr"] == round(res["q3"] - res["q1"], 2)
        assert res["lower_bound"] == round(res["q1"] - 1.5 * res["iqr"], 2)
        assert res["upper_bound"] == round(res["q3"] + 1.5 * res["iqr"], 2)


def test_detect_all_outliers_extreme_value(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    outliers = detect_all_outliers(enriched_df)

    # Unit_Price of 700000.0 in sample fixture is an extreme outlier
    assert outliers["Unit_Price"]["outlier_count"] >= 1
