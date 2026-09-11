"""
Effect size calculations: Cohen's d and practical significance classification.
"""

from typing import Optional

def compute_cohens_d(sample_mean: float, mu_0: float, sample_std: float) -> dict:
    if sample_std <= 0:
        raise ValueError("Sample standard deviation must be strictly positive.")
        
    diff = sample_mean - mu_0
    d = diff / sample_std
    abs_d = abs(d)
    
    if abs_d < 0.20:
        tier = "Negligible"
        desc = "Effect is negligible in real-world magnitude."
    elif abs_d < 0.50:
        tier = "Small"
        desc = "Small practical effect; noticeable only with rigorous tracking."
    elif abs_d < 0.80:
        tier = "Medium"
        desc = "Substantial medium effect observable in operations."
    else:
        tier = "Large"
        desc = "Large practical effect with clear business impact."
        
    return {
        "difference": diff,
        "cohens_d": d,
        "abs_cohens_d": abs_d,
        "magnitude": tier,
        "description": desc
    }
