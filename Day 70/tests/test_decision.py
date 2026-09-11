"""
Unit tests for decision engine reconciling p-value, critical value, and CI rules.
"""

try:
    from app.config import AlternativeType
    from app.decision import evaluate_decisions
except (ImportError, ModuleNotFoundError):
    from config import AlternativeType
    from decision import evaluate_decisions

def test_evaluate_decisions_reject_two_sided():
    dec = evaluate_decisions(
        p_value=0.01,
        test_statistic=2.58,
        critical_value=1.96,
        alternative=AlternativeType.TWO_SIDED,
        alpha=0.05,
        ci_lower=102.0,
        ci_upper=108.0,
        null_val=100.0
    )
    assert dec["reject_null"] is True
    assert "Reject" in dec["decision_pvalue"]
    assert "Reject" in dec["decision_critical"]
    assert "Reject" in dec["decision_ci"]
    assert dec["methods_agree"] is True

def test_evaluate_decisions_fail_to_reject():
    dec = evaluate_decisions(
        p_value=0.15,
        test_statistic=1.45,
        critical_value=1.96,
        alternative=AlternativeType.TWO_SIDED,
        alpha=0.05,
        ci_lower=98.0,
        ci_upper=104.0,
        null_val=100.0
    )
    assert dec["reject_null"] is False
    assert "Fail" in dec["decision_pvalue"]
    assert "Fail" in dec["decision_critical"]
    assert "Fail" in dec["decision_ci"]
    assert dec["methods_agree"] is True
