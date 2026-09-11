"""
Unit tests for experimentation metrics and SRM checks.
"""

import math
import pandas as pd
import pytest

try:
    from app.metrics import (
        compute_conversion_metrics,
        compute_financial_metrics,
        compute_guardrail_metrics,
        check_sample_ratio_mismatch
    )
except (ImportError, ModuleNotFoundError):
    from metrics import (
        compute_conversion_metrics,
        compute_financial_metrics,
        compute_guardrail_metrics,
        check_sample_ratio_mismatch
    )

def test_compute_conversion_metrics():
    ctrl = pd.DataFrame({"converted": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]})  # 1/10 = 10%
    trt = pd.DataFrame({"converted": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})   # 2/10 = 20%
    m = compute_conversion_metrics(ctrl, trt)
    assert m["n_control"] == 10
    assert m["n_treatment"] == 10
    assert m["rate_control"] == 0.10
    assert m["rate_treatment"] == 0.20
    assert math.isclose(m["absolute_lift"], 0.10)
    assert math.isclose(m["relative_lift_pct"], 100.0)

def test_compute_financial_metrics():
    ctrl = pd.DataFrame({"revenue": [0.0, 50.0, 0.0, 100.0]})  # ARPU = 150/4 = 37.5, AOV = 75
    trt = pd.DataFrame({"revenue": [0.0, 60.0, 80.0, 160.0]})  # ARPU = 300/4 = 75.0, AOV = 100
    fin = compute_financial_metrics(ctrl, trt)
    assert fin["arpu_control"] == 37.50
    assert fin["arpu_treatment"] == 75.00
    assert fin["arpu_lift"] == 37.50
    assert fin["aov_control"] == 75.00
    assert fin["aov_treatment"] == 100.00

def test_compute_guardrail_metrics():
    ctrl = pd.DataFrame({"bounce": [1, 0, 0, 0], "refunded": [0, 0, 0, 0], "session_duration": [10, 100, 110, 120]})
    trt = pd.DataFrame({"bounce": [1, 1, 0, 0], "refunded": [0, 1, 0, 0], "session_duration": [10, 20, 150, 180]})
    g = compute_guardrail_metrics(ctrl, trt)
    assert g["bounce_rate_control"] == 0.25
    assert g["bounce_rate_treatment"] == 0.50
    assert g["bounce_delta"] == 0.25
    assert g["refund_rate_treatment"] == 0.25

def test_check_sample_ratio_mismatch_healthy():
    srm = check_sample_ratio_mismatch(n_control=5020, n_treatment=4980)
    assert srm["srm_detected"] is False
    assert srm["p_value"] > 0.05

def test_check_sample_ratio_mismatch_flagged():
    # 4500 vs 5500 is massive SRM on 10,000 users (p < 1e-10)
    srm = check_sample_ratio_mismatch(n_control=4500, n_treatment=5500)
    assert srm["srm_detected"] is True
    assert srm["p_value"] < 0.001
