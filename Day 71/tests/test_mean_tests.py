"""
Unit tests for two-sample mean comparison functions.
"""

import math
import numpy as np
import pytest
from scipy import stats

try:
    from app.mean_tests import independent_ttest, paired_ttest
except (ImportError, ModuleNotFoundError):
    from mean_tests import independent_ttest, paired_ttest

def test_independent_ttest_welch_vs_scipy(continuous_groups):
    a, b = continuous_groups
    res = independent_ttest(a, b, equal_var=False, alpha=0.05)
    scipy_res = stats.ttest_ind(b, a, equal_var=False)
    
    assert math.isclose(res["t_statistic"], scipy_res.statistic, rel_tol=1e-5)
    assert math.isclose(res["p_value"], scipy_res.pvalue, rel_tol=1e-5)
    assert res["test_type"] == "Welch's Two-Sample t-test"

def test_independent_ttest_students_equal_var(continuous_groups):
    a, b = continuous_groups
    res = independent_ttest(a, b, equal_var=True, alpha=0.05)
    scipy_res = stats.ttest_ind(b, a, equal_var=True)
    assert math.isclose(res["t_statistic"], scipy_res.statistic, rel_tol=1e-5)
    assert math.isclose(res["p_value"], scipy_res.pvalue, rel_tol=1e-5)

def test_independent_ttest_too_few_elements():
    with pytest.raises(ValueError, match="at least 2 valid numeric observations"):
        independent_ttest([1.0], [2.0, 3.0])

def test_paired_ttest_vs_scipy(paired_data):
    before, after = paired_data
    res = paired_ttest(before, after, alpha=0.05)
    scipy_res = stats.ttest_rel(after, before)
    assert math.isclose(res["t_statistic"], scipy_res.statistic, rel_tol=1e-5)
    assert math.isclose(res["p_value"], scipy_res.pvalue, rel_tol=1e-5)
    assert res["test_type"] == "Paired Two-Sample t-test"

def test_paired_ttest_mismatched_lengths():
    with pytest.raises(ValueError, match="identical length"):
        paired_ttest([1, 2, 3], [1, 2])

def test_independent_ttest_equal_means():
    data = [10.0, 20.0, 30.0, 40.0]
    res = independent_ttest(data, data)
    assert math.isclose(res["t_statistic"], 0.0)
    assert math.isclose(res["difference"], 0.0)
    assert res["reject_null"] is False

def test_independent_ttest_nan_filtering():
    a = [10.0, np.nan, 20.0, 30.0]
    b = [25.0, 35.0, np.nan, 45.0]
    res = independent_ttest(a, b)
    assert res["n_control"] == 3
    assert res["n_treatment"] == 3
