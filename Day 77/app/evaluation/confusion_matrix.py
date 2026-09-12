import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_true, y_pred, labels=None) -> dict:
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    # Identify most confused pair (off-diagonal maximum in normalized matrix)
    n = len(labels)
    confused_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                confused_pairs.append({
                    "True_Class": labels[i],
                    "Predicted_Class": labels[j],
                    "Count": int(cm[i, j]),
                    "Rate": float(cm_norm[i, j])
                })
    confused_df = pd.DataFrame(confused_pairs).sort_values(by="Count", ascending=False)
    
    res = {
        "labels": labels,
        "matrix": cm,
        "normalized_matrix": cm_norm,
        "dataframe": pd.DataFrame(cm, index=labels, columns=labels),
        "normalized_df": pd.DataFrame(cm_norm, index=labels, columns=labels),
        "confused_pairs": confused_df
    }
    
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        res["TN"] = int(tn)
        res["FP"] = int(fp)
        res["FN"] = int(fn)
        res["TP"] = int(tp)
        
    return res
