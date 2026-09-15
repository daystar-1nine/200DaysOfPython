from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 300.0,
    cost_fn: float = 2000.0
) -> Dict[str, float]:
    """Calculate total business cost given FP (wasted retention offer) and FN (lost customer)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_fp_cost = fp * cost_fp
    total_fn_cost = fn * cost_fn
    total_cost = total_fp_cost + total_fn_cost
    avg_cost_per_customer = total_cost / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        'FP_Count': int(fp),
        'FN_Count': int(fn),
        'TP_Count': int(tp),
        'TN_Count': int(tn),
        'FP_Cost': float(total_fp_cost),
        'FN_Cost': float(total_fn_cost),
        'Total_Cost': float(total_cost),
        'Avg_Cost_Per_Customer': float(avg_cost_per_customer)
    }

def sweep_cost_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 300.0,
    cost_fn: float = 2000.0
) -> pd.DataFrame:
    """Evaluate business cost across thresholds 0.05 to 0.95."""
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        res = compute_business_cost(y_true, y_pred, cost_fp, cost_fn)
        records.append({
            'Threshold': float(t),
            'FP_Count': res['FP_Count'],
            'FN_Count': res['FN_Count'],
            'Total_Cost': res['Total_Cost'],
            'Avg_Cost_Per_Customer': res['Avg_Cost_Per_Customer']
        })
        
    return pd.DataFrame(records)

def find_minimum_cost_threshold(cost_df: pd.DataFrame) -> Tuple[float, float]:
    """Find threshold that minimizes total business cost."""
    min_idx = cost_df['Total_Cost'].idxmin()
    min_t = float(cost_df.loc[min_idx, 'Threshold'])
    min_cost = float(cost_df.loc[min_idx, 'Total_Cost'])
    return min_t, min_cost
