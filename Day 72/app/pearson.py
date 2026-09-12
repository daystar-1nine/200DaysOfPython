"""
Day 72 — Pearson Correlation Engine
Computes Pearson r, test statistic t, two-tailed p-value, and Fisher z-transformed confidence interval.
"""
from typing import Dict, Any
import numpy as np
from scipy import stats

try:
    from app.config import StrengthTier, DirectionType, classify_strength
except ImportError:
    from config import StrengthTier, DirectionType, classify_strength

def compute_pearson_detail(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    n = len(x)
    if n < 3:
        raise ValueError("Sample size must be at least 3 for Pearson hypothesis testing.")
        
    r, p_val = stats.pearsonr(x, y)
    r = float(r)
    p_val = float(p_val)
    
    # Degrees of freedom and t-statistic
    df = n - 2
    if abs(r) >= 1.0:
        t_stat = np.inf if r > 0 else -np.inf
        p_val = 0.0
        ci_lower = r
        ci_upper = r
    else:
        t_stat = float(r * np.sqrt(df / (1.0 - r ** 2)))
        # Fisher z-transformation for CI
        z = np.arctanh(r)
        se_z = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lower = float(np.tanh(z - z_crit * se_z))
        ci_upper = float(np.tanh(z + z_crit * se_z))
        
    direction = DirectionType.POSITIVE if r > 1e-6 else (DirectionType.NEGATIVE if r < -1e-6 else DirectionType.ZERO)
    strength = classify_strength(abs(r))
    
    return {
        "r": r,
        "p_value": p_val,
        "t_statistic": t_stat,
        "df": df,
        "sample_size": n,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "direction": direction.value,
        "strength": strength.value,
        "is_significant": bool(p_val < alpha)
    }
