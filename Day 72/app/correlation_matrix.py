"""
Day 72 — Correlation Matrix Engine
Computes Pearson, Spearman, and Difference matrices and generates comprehensive pairwise tables.
"""
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

try:
    from app.pearson import compute_pearson_detail
    from app.spearman import compute_spearman_detail
    from app.config import AppConfig
except ImportError:
    from pearson import compute_pearson_detail
    from spearman import compute_spearman_detail
    from config import AppConfig

def compute_correlation_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pearson_mat = df.corr(method="pearson")
    spearman_mat = df.corr(method="spearman")
    diff_mat = (pearson_mat - spearman_mat).abs()
    return pearson_mat, spearman_mat, diff_mat

def compute_all_pairwise_results(df: pd.DataFrame, config: AppConfig = None) -> pd.DataFrame:
    if config is None:
        config = AppConfig()
        
    cols = df.columns.tolist()
    records: List[Dict[str, Any]] = []
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            x = df[col1].values
            y = df[col2].values
            
            p_res = compute_pearson_detail(x, y, alpha=config.ALPHA)
            s_res = compute_spearman_detail(x, y, alpha=config.ALPHA)
            
            abs_diff = abs(p_res["r"] - s_res["rho"])
            is_strong = abs(p_res["r"]) >= config.STRONG_THRESHOLD
            is_multicollinear = abs(p_res["r"]) >= config.MULTICOLLINEARITY_THRESHOLD
            is_divergent = abs_diff >= config.DIVERGENCE_THRESHOLD
            
            records.append({
                "Variable_1": col1,
                "Variable_2": col2,
                "Pearson_r": round(p_res["r"], 4),
                "Pearson_p": p_res["p_value"],
                "CI_Lower": round(p_res["ci_lower"], 4),
                "CI_Upper": round(p_res["ci_upper"], 4),
                "Spearman_rho": round(s_res["rho"], 4),
                "Spearman_p": s_res["p_value"],
                "Abs_Diff": round(abs_diff, 4),
                "Strength": p_res["strength"],
                "Direction": p_res["direction"],
                "Is_Significant": p_res["is_significant"],
                "Is_Strong": is_strong,
                "Is_Multicollinear": is_multicollinear,
                "Is_Divergent": is_divergent
            })
            
    return pd.DataFrame(records)
