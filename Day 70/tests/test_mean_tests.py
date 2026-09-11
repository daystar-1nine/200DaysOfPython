"""
Unit tests for 1-sample t-test and Z-test implementations.
"""

import math
import numpy as np
import pytest
from scipy import stats

try:
    from app.tests_mean import one_sample_t_test, one_sample_z_test
except (ImportError, ModuleNotFoundError):
    from tests_mean import one_sample_t_test, one_sample_z_test

def test_one_sample_t_test_matches_scipy():
    np.random.seed(42)
    sample = np.random.normal(loc=55.0, scale=8.0, size=40)
    mu_0 = 50.0
    
    res = one_sample_t_test(sample, mu_0=mu_0, alpha=0.05, alternative="two-sided")
    scipy_res = stats.ttest_1samp(sample, popmean=mu_0)
    
    assert math.isclose(res["test_statistic"], scipy_res.statistic, rel_tol=1e-6)
    assert math.isclose(res["p_value"], scipy_res.pvalue, rel_tol=1e-6)
    assert res["reject_null"] is True

def test_one_sample_t_test_greater():
    data = [12.0, 14.0, 15.0, 16.0, 18.0]  # mean = 15.0
    res = one_sample_t_test(data, mu_0=10.0, alpha=0.05, alternative="greater")
    assert res["test_statistic"] > 0
    assert res["p_value"] < 0.05
    assert res["reject_null"] is True

def test_one_sample_t_test_less():
    data = [8.0, 9.0, 10.0, 9.5, 8.5]  # mean = 9.0
    res = one_sample_t_test(data, mu_0=12.0, alpha=0.05, alternative="less")
    assert res["test_statistic"] < 0
    assert res["p_value"] < 0.05
    assert res["reject_null"] is True

def test_one_sample_t_test_fail_to_reject():
    data = [99.5, 100.2, 99.8, 100.1, 100.4]
    res = one_sample_t_test(data, mu_0=100.0, alpha=0.05, alternative="two-sided")
    assert res["p_value"] > 0.05
    assert res["reject_null"] is False

def test_one_sample_z_test_two_sided():
    # sample_mean = 105, mu_0 = 100, sigma = 10, n = 25 -> SE = 10/5 = 2.0 -> z = (105-100)/2 = 2.5
    res = one_sample_z_test(sample_mean=105.0, n=25, mu_0=100.0, sigma=10.0, alpha=0.05, alternative="two-sided")
    assert math.isclose(res["test_statistic"], 2.5)
    assert math.isclose(res["critical_value"], 1.959963984540054)
    assert res["reject_null"] is True

def test_one_sample_z_test_one_sided():
    res = one_sample_z_test(sample_mean=105.0, n=25, mu_0=100.0, sigma=10.0, alpha=0.05, alternative="greater")
    assert math.isclose(res["critical_value"], 1.6448536269514722)
    assert res["reject_null"] is True

def test_one_sample_z_test_invalid_inputs():
    with pytest.raises(ValueError):
        one_sample_z_test(sample_mean=100.0, n=0, mu_0=100.0, sigma=5.0)
    with pytest.raises(ValueError):
        one_sample_z_test(sample_mean=100.0, n=25, mu_0=100.0, sigma=-1.0)
