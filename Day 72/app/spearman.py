"""
Day 72 — Spearman Rank Correlation Engine
Computes Spearman rho, rank transformation, and two-tailed p-value.
"""
from typing import Dict, Any
import numpy as np
from scipy import stats

try:
    from app.config import StrengthTier, DirectionType, classify_strength
except ImportError:
    from config import StrengthTier, DirectionType, classify_strength

def compute_spearman_detail(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    n = len(x)
    if n < 3:
        raise ValueError("Sample size must be at least 3 for Spearman correlation.")
        
    res = stats.spearmanr(x, y)
    rho = float(res.statistic)
    p_val = float(res.pvalue)
    
    direction = DirectionType.POSITIVE if rho > 1e-6 else (DirectionType.NEGATIVE if rho < -1e-6 else DirectionType.ZERO)
    strength = classify_strength(abs(rho))
    
    return {
        "rho": rho,
        "p_value": p_val,
        "sample_size": n,
        "direction": direction.value,
        "strength": strength.value,
        "is_significant": bool(p_val < alpha)
    }
