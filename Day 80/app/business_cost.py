from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 300.0,
    cost_fn: float = 2000.0
) -> Dict[str, float]:
    """Compute financial loss based on asymmetric false positive and false negative costs."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fp_cost = float(fp * cost_fp)
    fn_cost = float(fn * cost_fn)
    total_cost = fp_cost + fn_cost
    avg_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'TN': int(tn),
        'FP_Cost': fp_cost,
        'FN_Cost': fn_cost,
        'Total_Cost': total_cost,
        'Avg_Cost_Per_Customer': avg_cost
    }

def sweep_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 300.0,
    cost_fn: float = 2000.0
) -> pd.DataFrame:
    """Evaluate financial business cost across decision thresholds."""
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        res = compute_total_cost(y_true, y_pred, cost_fp, cost_fn)
        records.append({
            'Threshold': float(t),
            'FP': res['FP'],
            'FN': res['FN'],
            'Total_Cost': res['Total_Cost'],
            'Avg_Cost': res['Avg_Cost_Per_Customer']
        })
        
    return pd.DataFrame(records)

def find_optimal_cost_threshold(cost_df: pd.DataFrame) -> Tuple[float, float]:
    """Find threshold that minimizes total business cost."""
    min_idx = cost_df['Total_Cost'].idxmin()
    return float(cost_df.loc[min_idx, 'Threshold']), float(cost_df.loc[min_idx, 'Total_Cost'])
