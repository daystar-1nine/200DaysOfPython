"""
Unit tests for correlation matrix engine.
"""
import pytest
import pandas as pd
import numpy as np
from app.correlation_matrix import compute_correlation_matrices, compute_all_pairwise_results

def test_correlation_matrices_properties(sample_numeric_df):
    p_mat, s_mat, d_mat = compute_correlation_matrices(sample_numeric_df)
    assert p_mat.shape == (4, 4)
    assert s_mat.shape == (4, 4)
    assert d_mat.shape == (4, 4)
    # Diagonal elements must be 1.0 for r and rho, 0.0 for diff
    for col in sample_numeric_df.columns:
        assert np.isclose(p_mat.loc[col, col], 1.0)
        assert np.isclose(s_mat.loc[col, col], 1.0)
        assert np.isclose(d_mat.loc[col, col], 0.0)

def test_all_pairwise_records_count(sample_numeric_df):
    res_df = compute_all_pairwise_results(sample_numeric_df)
    # With 4 columns, total pairs = 4 * 3 / 2 = 6
    assert len(res_df) == 6
    required_cols = [
        "Variable_1", "Variable_2", "Pearson_r", "Pearson_p",
        "Spearman_rho", "Spearman_p", "Abs_Diff", "Is_Significant"
    ]
    for c in required_cols:
        assert c in res_df.columns
