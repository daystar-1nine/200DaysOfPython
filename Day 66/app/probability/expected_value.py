"""
Expected value and variance calculations for discrete random variables.
"""
from typing import Sequence
import numpy as np

def calculate_expected_value(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """
    Calculate E[X] = sum(x_i * p_i).
    """
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have identical lengths.")
    if len(values) == 0:
        raise ValueError("Input sequences cannot be empty.")
    
    prob_arr = np.array(probabilities, dtype=float)
    val_arr = np.array(values, dtype=float)
    
    if np.any(prob_arr < 0):
        raise ValueError("Probabilities cannot be negative.")
    if not np.isclose(np.sum(prob_arr), 1.0, atol=1e-3):
        raise ValueError(f"Probabilities must sum to 1.0 (current sum: {np.sum(prob_arr):.4f}).")
        
    return float(np.sum(val_arr * prob_arr))

def calculate_variance_discrete(values: Sequence[float], probabilities: Sequence[float]) -> dict:
    """
    Calculate Var(X) = E[X^2] - (E[X])^2 and SD(X) = sqrt(Var(X)).
    """
    ev = calculate_expected_value(values, probabilities)
    val_arr = np.array(values, dtype=float)
    prob_arr = np.array(probabilities, dtype=float)
    
    ev_sq = float(np.sum((val_arr ** 2) * prob_arr))
    variance = max(0.0, ev_sq - (ev ** 2))
    std_dev = np.sqrt(variance)
    
    return {
        "expected_value": round(ev, 4),
        "expected_value_sq": round(ev_sq, 4),
        "variance": round(variance, 4),
        "std_dev": round(float(std_dev), 4)
    }
