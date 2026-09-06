"""
Basic theoretical and empirical probability functions.
"""
from typing import Collection, Any

def theoretical_probability(favorable_outcomes: int, sample_space_size: int) -> float:
    """Calculate P(E) = n(E) / n(S)."""
    if sample_space_size <= 0:
        raise ValueError("Sample space size must be strictly positive.")
    if favorable_outcomes < 0 or favorable_outcomes > sample_space_size:
        raise ValueError("Favorable outcomes must be between 0 and sample_space_size.")
    return favorable_outcomes / sample_space_size

def empirical_probability(occurrences: int, total_trials: int) -> float:
    """Calculate empirical probability P_hat(E) = count / total_trials."""
    if total_trials <= 0:
        raise ValueError("Total trials must be strictly positive.")
    if occurrences < 0 or occurrences > total_trials:
        raise ValueError("Occurrences must be between 0 and total_trials.")
    return occurrences / total_trials

def complement_probability(p_event: float) -> float:
    """Calculate P(not E) = 1 - P(E)."""
    if not 0.0 <= p_event <= 1.0:
        raise ValueError("Probability must be in [0, 1].")
    return 1.0 - p_event

def union_probability(p_a: float, p_b: float, p_intersection: float = 0.0) -> float:
    """Calculate P(A or B) = P(A) + P(B) - P(A and B)."""
    if not (0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0 and 0.0 <= p_intersection <= 1.0):
        raise ValueError("All probabilities must be in [0, 1].")
    if p_intersection > min(p_a, p_b):
        raise ValueError("P(A and B) cannot exceed P(A) or P(B).")
    res = p_a + p_b - p_intersection
    return min(1.0, max(0.0, res))
