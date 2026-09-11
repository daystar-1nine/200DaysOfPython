"""
Unit tests for two-proportion pooled Z-test.
"""

import math
import pytest
from scipy import stats

try:
    from app.proportion_tests import two_proportion_ztest
except (ImportError, ModuleNotFoundError):
    from proportion_tests import two_proportion_ztest

def test_two_proportion_ztest_two_sided():
    # Control: 500/10,000 = 0.05, Treatment: 600/10,000 = 0.06
    res = two_proportion_ztest(500, 10000, 600, 10000, alpha=0.05, alternative="two-sided")
    assert res["rate_control"] == 0.05
    assert res["rate_treatment"] == 0.06
    assert math.isclose(res["absolute_lift"], 0.01)
    assert math.isclose(res["relative_lift_pct"], 20.0)
    assert res["z_statistic"] > 2.5
    assert res["p_value"] < 0.01
    assert res["reject_null"] is True

def test_two_proportion_ztest_greater_alternative():
    res = two_proportion_ztest(500, 10000, 600, 10000, alpha=0.05, alternative="greater")
    assert res["reject_null"] is True
    assert res["p_value"] < 0.01

def test_two_proportion_ztest_less_alternative():
    # Treatment is higher, so testing if Treatment is LESS should yield high p-value
    res = two_proportion_ztest(500, 10000, 600, 10000, alpha=0.05, alternative="less")
    assert res["reject_null"] is False
    assert res["p_value"] > 0.99

def test_two_proportion_ztest_invalid_alternative():
    with pytest.raises(ValueError, match="Invalid alternative"):
        two_proportion_ztest(50, 100, 60, 100, alternative="unequal")
