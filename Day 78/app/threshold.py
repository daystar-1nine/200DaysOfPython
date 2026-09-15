import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_thresholds(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
        
    records = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        records.append({
            "Threshold": round(float(t), 2),
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1_Score": round(float(f), 4)
        })
    return pd.DataFrame(records)

def find_optimal_threshold_f1(threshold_df: pd.DataFrame) -> float:
    best_row = threshold_df.loc[threshold_df["F1_Score"].idxmax()]
    return float(best_row["Threshold"])
