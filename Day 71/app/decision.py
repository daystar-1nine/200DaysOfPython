"""
Multi-criteria decision engine evaluating statistical significance, practical MDE, and guardrails.
"""

from dataclasses import dataclass

@dataclass
class DecisionOutcome:
    launch_recommendation: str
    is_statistically_significant: bool
    is_practically_meaningful: bool
    is_positive_lift: bool
    guardrails_passed: bool
    rationale: str

def evaluate_ab_test_decision(
    p_value: float,
    alpha: float,
    relative_lift_pct: float,
    mde_pct: float,
    bounce_delta: float,
    max_allowed_bounce_delta: float,
    refund_delta: float,
    max_allowed_refund_delta: float,
    srm_detected: bool = False
) -> DecisionOutcome:
    if srm_detected:
        return DecisionOutcome(
            launch_recommendation="HARD STOP / INVALIDATE EXPERIMENT",
            is_statistically_significant=False,
            is_practically_meaningful=False,
            is_positive_lift=False,
            guardrails_passed=False,
            rationale="Sample Ratio Mismatch (SRM) detected! Traffic allocation was corrupted. Invalidate test and fix assignment bug."
        )

    is_sig = p_value < alpha
    is_pos = relative_lift_pct > 0
    is_practical = relative_lift_pct >= mde_pct
    
    # Guardrail checks
    bounce_ok = bounce_delta <= max_allowed_bounce_delta
    refund_ok = refund_delta <= max_allowed_refund_delta
    guardrails_ok = bounce_ok and refund_ok
    
    if is_sig and is_pos and is_practical and guardrails_ok:
        decision = "RECOMMEND TREATMENT LAUNCH"
        reason = f"Statistically significant lift (+{relative_lift_pct:.2f}%) exceeds MDE ({mde_pct:.2f}%) with healthy guardrails."
    elif is_sig and not is_pos:
        decision = "DO NOT LAUNCH (STATISTICALLY SIGNIFICANT REGRESSION)"
        reason = f"Treatment significantly underperformed Control (Relative lift: {relative_lift_pct:.2f}%)."
    elif is_sig and is_pos and not guardrails_ok:
        decision = "REJECT TREATMENT (GUARDRAIL FAILURE)"
        reason = f"Conversion lift detected (+{relative_lift_pct:.2f}%), but guardrail metrics failed (Bounce delta: {bounce_delta:+.2%}, Refund delta: {refund_delta:+.2%})."
    elif is_sig and is_pos and not is_practical:
        decision = "CONTINUE INVESTIGATION (STATISTICALLY SIGNIFICANT BUT TRIVIAL EFFECT)"
        reason = f"Lift is statistically significant but below business Minimum Detectable Effect ({relative_lift_pct:.2f}% < {mde_pct:.2f}%)."
    else:
        decision = "INCONCLUSIVE (CONTINUE EXPERIMENT)"
        reason = f"Insufficient evidence to reject null hypothesis at alpha={alpha} (p={p_value:.4f})."
        
    return DecisionOutcome(
        launch_recommendation=decision,
        is_statistically_significant=is_sig,
        is_practically_meaningful=is_practical,
        is_positive_lift=is_pos,
        guardrails_passed=guardrails_ok,
        rationale=reason
    )
