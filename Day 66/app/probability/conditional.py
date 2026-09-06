"""
Conditional probability and statistical independence calculations.
"""

def conditional_probability(p_intersection: float, p_given: float) -> float:
    """Calculate P(A | B) = P(A and B) / P(B)."""
    if not (0.0 <= p_intersection <= 1.0 and 0.0 < p_given <= 1.0):
        raise ValueError("Invalid probability values; p_given must be > 0.")
    if p_intersection > p_given:
        raise ValueError("P(A and B) cannot exceed P(B).")
    return p_intersection / p_given

def joint_probability(p_given: float, p_conditional: float) -> float:
    """Calculate P(A and B) = P(B) * P(A | B)."""
    if not (0.0 <= p_given <= 1.0 and 0.0 <= p_conditional <= 1.0):
        raise ValueError("Probabilities must be in [0, 1].")
    return p_given * p_conditional

def independence_test(p_a: float, p_b: float, p_intersection: float, tolerance: float = 1e-4) -> dict:
    """
    Test whether events A and B are independent:
    P(A and B) == P(A) * P(B).
    """
    expected_joint = p_a * p_b
    diff = abs(p_intersection - expected_joint)
    is_independent = diff <= tolerance
    return {
        "p_a": p_a,
        "p_b": p_b,
        "p_joint_actual": p_intersection,
        "p_joint_expected_if_independent": round(expected_joint, 6),
        "absolute_difference": round(diff, 6),
        "is_independent": is_independent
    }
