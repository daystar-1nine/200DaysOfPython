"""
Decision logic engine synthesizing P-value, Critical Value, and Confidence Interval rules.
"""

try:
    from app.config import AlternativeType
    from app.confidence_intervals import check_ci_contains_null
except (ImportError, ModuleNotFoundError):
    from config import AlternativeType
    from confidence_intervals import check_ci_contains_null

def evaluate_decisions(
    p_value: float,
    test_statistic: float,
    critical_value: float,
    alternative: AlternativeType,
    alpha: float,
    ci_lower: float,
    ci_upper: float,
    null_val: float
) -> dict:
    # 1. P-value rule
    reject_p = p_value <= alpha
    dec_p = "Reject H0 (Statistically Significant)" if reject_p else "Fail to Reject H0"
    
    # 2. Critical value rule
    if alternative == AlternativeType.TWO_SIDED:
        reject_crit = abs(test_statistic) >= abs(critical_value)
    elif alternative == AlternativeType.GREATER:
        reject_crit = test_statistic >= critical_value
    else:
        reject_crit = test_statistic <= critical_value
    dec_crit = "Reject H0 (Critical Region)" if reject_crit else "Fail to Reject H0"
    
    # 3. Confidence interval rule (for two-sided tests)
    contains_null = check_ci_contains_null(ci_lower, ci_upper, null_val)
    if alternative == AlternativeType.TWO_SIDED:
        reject_ci = not contains_null
        dec_ci = "Reject H0 (Null outside CI)" if reject_ci else "Fail to Reject H0 (Null inside CI)"
    else:
        reject_ci = reject_p
        dec_ci = "N/A (One-tailed test)"
        
    # Synthesis
    is_consistent = (reject_p == reject_crit)
    
    return {
        "reject_null": reject_p,
        "decision_pvalue": dec_p,
        "decision_critical": dec_crit,
        "decision_ci": dec_ci,
        "methods_agree": is_consistent
    }
