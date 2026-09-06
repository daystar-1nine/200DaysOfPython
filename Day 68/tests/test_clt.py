"""
Tests for CLTAnalyzer.
"""
import pytest
import numpy as np
from app.clt import CLTAnalyzer

def test_clt_convergence_standard_error_decay(skewed_population):
    analyzer = CLTAnalyzer(skewed_population, seed=42)
    sample_sizes = [10, 40, 90]
    df = analyzer.evaluate_convergence(sample_sizes, n_samples=2_000)
    
    # Check that SE decreases as sample size increases
    ses = df["theo_se"].values
    assert ses[0] > ses[1] > ses[2]
    
    # Quadrupling n from 10 to 40 roughly halves SE
    # Ratio ~ sqrt(40/10) = 2.0
    ratio = ses[0] / ses[1]
    assert np.isclose(ratio, 2.0, atol=0.1)

def test_clt_skewness_reduction(skewed_population):
    analyzer = CLTAnalyzer(skewed_population, seed=42)
    sample_sizes = [2, 20, 200]
    df = analyzer.evaluate_convergence(sample_sizes, n_samples=2_000)
    
    skews = df["skewness"].values
    # Skewness should attenuate towards 0 as sample size increases
    assert abs(skews[0]) > abs(skews[1]) > abs(skews[2])


def test_clt_dataframe_schema(uniform_population):
    analyzer = CLTAnalyzer(uniform_population, seed=42)
    df = analyzer.evaluate_convergence([10, 30], n_samples=500)
    
    expected_cols = [
        "sample_size", "emp_mean", "theo_se", "emp_se",
        "se_error", "skewness", "p_normaltest", "is_approx_normal"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
