"""
Unit tests for one-sample proportion tests.
"""

import math
import pytest

try:
    from app.tests_proportion import one_sample_proportion_test
except (ImportError, ModuleNotFoundError):
    from tests_proportion import one_sample_proportion_test

def test_proportion_test_significant():
    # 130 successes out of 1000, testing p_0 = 0.10
    # SE0 = sqrt(0.1 * 0.9 / 1000) = sqrt(0.00009) = 0.0094868
    # z = (0.13 - 0.10) / 0.0094868 = 3.16227
    res = one_sample_proportion_test(successes=130, trials=1000, p_0=0.10, alpha=0.05, alternative="greater")
    assert math.isclose(res["sample_proportion"], 0.13)
    assert math.isclose(res["test_statistic"], 3.162277, rel_tol=1e-4)
    assert res["normality_assumption_met"] is True
    assert res["p_value"] < 0.001
    assert res["reject_null"] is True

def test_proportion_test_not_significant():
    res = one_sample_proportion_test(successes=102, trials=1000, p_0=0.10, alpha=0.05, alternative="two-sided")
    assert res["p_value"] > 0.05
    assert res["reject_null"] is False

def test_proportion_test_two_sided():
    res = one_sample_proportion_test(successes=120, trials=1000, p_0=0.10, alpha=0.05, alternative="two-sided")
    assert math.isclose(res["critical_value"], 1.95996, rel_tol=1e-4)
    assert res["reject_null"] is True

def test_proportion_test_normality_check():
    # n=10, p0=0.1 -> n*p0 = 1 < 10 -> normality invalid
    res = one_sample_proportion_test(successes=2, trials=10, p_0=0.10)
    assert res["normality_assumption_met"] is False
