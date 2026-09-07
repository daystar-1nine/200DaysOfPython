"""
Data validation module ensuring boundary conditions and data integrity.
"""
import numpy as np

def validate_numeric_array(data: np.ndarray | list[float], allow_empty: bool = False) -> np.ndarray:
    if data is None:
        raise ValueError("Data cannot be None.")
    try:
        arr = np.asarray(data, dtype=float)
    except Exception as e:
        raise ValueError(f"Data could not be converted to float array: {e}")
        
    if len(arr) == 0 and not allow_empty:
        raise ValueError("Data array cannot be empty.")
    if np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(arr)):
        raise ValueError("Data contains Infinite values.")
    return arr

def validate_confidence_level(confidence: float) -> float:
    if not isinstance(confidence, (int, float)):
        raise TypeError("Confidence level must be a numeric float.")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence level must be strictly between 0 and 1, got {confidence}.")
    return float(confidence)

def validate_sample_size(n: int) -> int:
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Sample size must be an integer, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"Sample size must be strictly positive, got {n}.")
    return int(n)

def validate_proportion_counts(successes: int, total: int) -> tuple[int, int]:
    if total <= 0:
        raise ValueError(f"Total observations must be strictly positive, got {total}.")
    if successes < 0:
        raise ValueError(f"Successes cannot be negative, got {successes}.")
    if successes > total:
        raise ValueError(f"Successes ({successes}) cannot exceed total observations ({total}).")
    return int(successes), int(total)

def validate_margin_of_error(me: float) -> float:
    if me <= 0.0:
        raise ValueError(f"Margin of error must be strictly positive, got {me}.")
    return float(me)
