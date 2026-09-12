"""
Day 73 - Exploratory Data Analysis Engine
Computes summary statistics and Pearson correlation between advertising spend and sales.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

def compute_eda(df: pd.DataFrame, feature_col: str = "Advertising_Spend", target_col: str = "Sales") -> Dict[str, Any]:
    x = df[feature_col].values
    y = df[target_col].values
    
    r_val, p_val = stats.pearsonr(x, y)
    
    return {
        "count": len(df),
        "mean_x": float(np.mean(x)),
        "std_x": float(np.std(x, ddof=1)),
        "min_x": float(np.min(x)),
        "max_x": float(np.max(x)),
        "mean_y": float(np.mean(y)),
        "std_y": float(np.std(y, ddof=1)),
        "min_y": float(np.min(y)),
        "max_y": float(np.max(y)),
        "pearson_r": float(r_val),
        "p_value": float(p_val)
    }
