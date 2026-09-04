"""
Unit Tests for app/analysis/statistics.py and app/analysis/outliers.py modules.
"""

from app.analysis.outliers import detect_outliers_iqr
from app.analysis.statistics import compute_correlations, compute_numerical_statistics
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_compute_numerical_statistics(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    stats_dict = compute_numerical_statistics(trans_df)

    assert "Revenue" in stats_dict
    assert "mean" in stats_dict["Revenue"]
    assert "q1" in stats_dict["Revenue"]
    assert "q3" in stats_dict["Revenue"]
    assert "iqr" in stats_dict["Revenue"]
    assert stats_dict["Revenue"]["q3"] >= stats_dict["Revenue"]["q1"]


def test_compute_correlations(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    corr_df, cov_df = compute_correlations(trans_df)

    assert "Revenue" in corr_df.columns
    assert "Profit" in corr_df.columns
    for col in corr_df.columns:
        assert corr_df.loc[col, col] == 1.0


def test_detect_outliers_iqr_bounds(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    outliers = detect_outliers_iqr(trans_df)

    for col, data in outliers.items():
        assert data["iqr"] == round(data["q3"] - data["q1"], 2)
        assert data["lower_bound"] == round(data["q1"] - 1.5 * data["iqr"], 2)
        assert data["upper_bound"] == round(data["q3"] + 1.5 * data["iqr"], 2)
