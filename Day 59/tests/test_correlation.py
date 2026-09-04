"""
Unit Tests for app/correlation.py module.
"""

import pandas as pd
from app.cleaner import clean_sales_data
from app.correlation import compute_correlation_matrix, compute_covariance_matrix
from app.transformer import compute_derived_metrics


def test_compute_correlation_matrix_shape(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    corr_df = compute_correlation_matrix(enriched_df)

    assert isinstance(corr_df, pd.DataFrame)
    assert "Revenue" in corr_df.columns
    assert "Profit" in corr_df.columns
    assert (corr_df.values >= -1.0).all() and (corr_df.values <= 1.0).all()


def test_compute_correlation_matrix_diagonal(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    corr_df = compute_correlation_matrix(enriched_df)

    for col in corr_df.columns:
        assert corr_df.loc[col, col] == 1.0


def test_compute_covariance_matrix_shape(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)
    cov_df = compute_covariance_matrix(enriched_df)

    assert isinstance(cov_df, pd.DataFrame)
    assert "Revenue" in cov_df.columns
