"""
Data validation and edge-case handling module.
"""

import numpy as np
import pandas as pd


def validate_and_clean_series(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Validates a pandas Series, cleans out NaNs and infs, and returns metadata.
    """
    meta = {
        "original_count": len(series),
        "nan_count": int(series.isna().sum()),
        "is_empty": False,
        "is_constant": False,
        "inf_count": 0
    }
    
    # Coerce to numeric
    numeric_s = pd.to_numeric(series, errors='coerce')
    
    # Check for infinite values
    inf_mask = np.isinf(numeric_s)
    meta["inf_count"] = int(inf_mask.sum())
    
    # Clean series: drop NaN and Inf
    clean_s = numeric_s[~numeric_s.isna() & ~inf_mask]
    
    if clean_s.empty:
        meta["is_empty"] = True
        return clean_s, meta
        
    if clean_s.nunique() <= 1:
        meta["is_constant"] = True
        
    return clean_s, meta
