"""
Input validation utilities for statistical tests.
"""

from typing import Union, Sequence
import numpy as np

def validate_numeric_array(data: Sequence[Union[int, float]], min_len: int = 2) -> np.ndarray:
    if data is None:
        raise ValueError("Data cannot be None.")
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < min_len:
        raise ValueError(f"Sample size must be at least {min_len}, got {len(arr)} valid numeric entries.")
    return arr

def validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"Alpha must be numeric, got {type(alpha).__name__}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance level alpha must be strictly between 0 and 1, got {alpha}")
    return float(alpha)

def validate_alternative(alt: str) -> str:
    alt_lower = str(alt).lower().strip()
    valid_alts = {"two-sided", "greater", "less"}
    if alt_lower not in valid_alts:
        raise ValueError(f"Alternative hypothesis must be one of {valid_alts}, got '{alt}'")
    return alt_lower

def validate_proportion(p: float, name: str = "Proportion") -> float:
    if not isinstance(p, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(p).__name__}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1, got {p}")
    return float(p)

def validate_counts(successes: int, trials: int) -> tuple[int, int]:
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError(f"Trials must be a positive integer, got {trials}")
    if not isinstance(successes, int) or successes < 0:
        raise ValueError(f"Successes must be a non-negative integer, got {successes}")
    if successes > trials:
        raise ValueError(f"Successes ({successes}) cannot exceed total trials ({trials})")
    return successes, trials
