"""
Day 72 — Outlier Detection & Correlation Leverage Engine
Identifies univariate and bivariate outliers and computes shift in Pearson r with/without outliers.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

def detect_univariate_outliers(series: pd.Series, k: float = 1.5) -> np.ndarray:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (series < lower_bound) | (series > upper_bound)

def evaluate_outlier_impact(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    x_all = df[col1].values
    y_all = df[col2].values
    r_all = float(stats.pearsonr(x_all, y_all).statistic)
    rho_all = float(stats.spearmanr(x_all, y_all).statistic)
    
    # Joint outlier mask based on individual IQRs
    mask_x = detect_univariate_outliers(df[col1])
    mask_y = detect_univariate_outliers(df[col2])
    outlier_mask = mask_x | mask_y
    n_outliers = int(outlier_mask.sum())
    
    if n_outliers > 0 and (len(df) - n_outliers >= 3):
        clean_df = df[~outlier_mask]
        r_clean = float(stats.pearsonr(clean_df[col1], clean_df[col2]).statistic)
        rho_clean = float(stats.spearmanr(clean_df[col1], clean_df[col2]).statistic)
    else:
        r_clean = r_all
        rho_clean = rho_all
        
    delta_r = abs(r_all - r_clean)
    has_high_leverage = delta_r >= 0.10
    
    return {
        "col1": col1,
        "col2": col2,
        "n_total": len(df),
        "n_outliers": n_outliers,
        "r_all": round(r_all, 4),
        "r_clean": round(r_clean, 4),
        "delta_r": round(delta_r, 4),
        "rho_all": round(rho_all, 4),
        "rho_clean": round(rho_clean, 4),
        "has_high_leverage": has_high_leverage
    }
