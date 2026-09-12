"""
Unit tests for covariance module.
"""
import pytest
import numpy as np
import pandas as pd
from app.covariance import compute_sample_covariance, compute_covariance_matrix

def test_sample_covariance_exact_match():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 5, 8, 10], dtype=float)
    cov_val = compute_sample_covariance(x, y)
    np_cov = np.cov(x, y, ddof=1)[0, 1]
    assert np.isclose(cov_val, np_cov)

def test_covariance_symmetry():
    x = np.random.normal(0, 1, 20)
    y = np.random.normal(0, 1, 20)
    cov_xy = compute_sample_covariance(x, y)
    cov_yx = compute_sample_covariance(y, x)
    assert np.isclose(cov_xy, cov_yx)

def test_covariance_matrix_shape():
    df = pd.DataFrame(np.random.rand(25, 4), columns=["A", "B", "C", "D"])
    cov_mat = compute_covariance_matrix(df)
    assert cov_mat.shape == (4, 4)
    assert np.isclose(cov_mat.loc["A", "A"], df["A"].var(ddof=1))

def test_covariance_unequal_lengths():
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(ValueError, match="equal length"):
        compute_sample_covariance(x, y)
