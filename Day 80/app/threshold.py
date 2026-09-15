from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Sweep classification decision thresholds from 0.05 to 0.95."""
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        records.append({
            'Threshold': float(t),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1': float(f1),
            'Accuracy': float(acc)
        })
        
    return pd.DataFrame(records)

def find_best_f1_threshold(df_thresh: pd.DataFrame) -> Tuple[float, float]:
    """Identify threshold maximizing F1 score."""
    best_idx = df_thresh['F1'].idxmax()
    return float(df_thresh.loc[best_idx, 'Threshold']), float(df_thresh.loc[best_idx, 'F1'])
