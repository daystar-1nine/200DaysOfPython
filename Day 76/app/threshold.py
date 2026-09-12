import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def apply_threshold(probabilities: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    return (probabilities >= threshold).astype(int)

def evaluate_thresholds(y_true: np.ndarray, probabilities: np.ndarray, thresholds: list = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
        
    results = []
    for t in thresholds:
        y_pred = apply_threshold(probabilities, t)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        results.append({
            'Threshold': t,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        })
        
    return pd.DataFrame(results)

def find_optimal_threshold(threshold_df: pd.DataFrame, metric: str = 'f1') -> float:
    metric_col = metric.capitalize()
    if metric_col not in threshold_df.columns:
        raise ValueError(f"Metric {metric} not found in threshold_df")
        
    optimal_idx = threshold_df[metric_col].idxmax()
    return threshold_df.loc[optimal_idx, 'Threshold']
