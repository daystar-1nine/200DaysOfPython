"""
Unit tests for multi-criteria experiment decision engine.
"""

import pytest

try:
    from app.decision import evaluate_ab_test_decision
except (ImportError, ModuleNotFoundError):
    from decision import evaluate_ab_test_decision

def test_decision_recommend_launch():
    res = evaluate_ab_test_decision(
        p_value=0.01,
        alpha=0.05,
        relative_lift_pct=12.0,
        mde_pct=8.0,
        bounce_delta=-0.02,
        max_allowed_bounce_delta=0.05,
        refund_delta=0.001,
        max_allowed_refund_delta=0.015,
        srm_detected=False
    )
    assert "RECOMMEND" in res.launch_recommendation
    assert res.is_statistically_significant is True
    assert res.guardrails_passed is True

def test_decision_guardrail_failure():
    res = evaluate_ab_test_decision(
        p_value=0.01,
        alpha=0.05,
        relative_lift_pct=15.0,
        mde_pct=8.0,
        bounce_delta=0.09,  # Bad bounce increase > 0.05
        max_allowed_bounce_delta=0.05,
        refund_delta=0.001,
        max_allowed_refund_delta=0.015,
        srm_detected=False
    )
    assert "GUARDRAIL FAILURE" in res.launch_recommendation
    assert res.guardrails_passed is False

def test_decision_srm_hard_stop():
    res = evaluate_ab_test_decision(
        p_value=0.001,
        alpha=0.05,
        relative_lift_pct=20.0,
        mde_pct=8.0,
        bounce_delta=0.0,
        max_allowed_bounce_delta=0.05,
        refund_delta=0.0,
        max_allowed_refund_delta=0.015,
        srm_detected=True  # Traffic corruption
    )
    assert "HARD STOP" in res.launch_recommendation

def test_decision_inconclusive():
    res = evaluate_ab_test_decision(
        p_value=0.15,
        alpha=0.05,
        relative_lift_pct=5.0,
        mde_pct=8.0,
        bounce_delta=0.0,
        max_allowed_bounce_delta=0.05,
        refund_delta=0.0,
        max_allowed_refund_delta=0.015,
        srm_detected=False
    )
    assert "INCONCLUSIVE" in res.launch_recommendation

def test_decision_significant_negative_regression():
    res = evaluate_ab_test_decision(
        p_value=0.001,
        alpha=0.05,
        relative_lift_pct=-10.0,
        mde_pct=8.0,
        bounce_delta=0.0,
        max_allowed_bounce_delta=0.05,
        refund_delta=0.0,
        max_allowed_refund_delta=0.015,
        srm_detected=False
    )
    assert "REGRESSION" in res.launch_recommendation

def test_decision_significant_below_mde():
    res = evaluate_ab_test_decision(
        p_value=0.01,
        alpha=0.05,
        relative_lift_pct=3.0,  # 3% is below 8% MDE
        mde_pct=8.0,
        bounce_delta=0.0,
        max_allowed_bounce_delta=0.05,
        refund_delta=0.0,
        max_allowed_refund_delta=0.015,
        srm_detected=False
    )
    assert "TRIVIAL EFFECT" in res.launch_recommendation
