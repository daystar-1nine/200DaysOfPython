import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from .confusion import compute_confusion_matrix_details

def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    thresholds: Optional[List[float]] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        
    records = []
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        _, cm_dict = compute_confusion_matrix_details(y_true, y_prob, threshold=t)
        
        records.append({
            "model": model_name,
            "threshold": round(t, 2),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positives": cm_dict["false_positives"],
            "false_negatives": cm_dict["false_negatives"]
        })
        
    df_thresh = pd.DataFrame(records)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_thresh.to_csv(save_path, index=False)
    return df_thresh
