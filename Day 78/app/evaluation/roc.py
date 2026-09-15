import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def compute_roc(y_true, y_prob) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val = float(roc_auc_score(y_true, y_prob))
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc_val,
        "optimal_threshold_youden": float(thresholds[best_idx]),
        "optimal_j": float(j_scores[best_idx])
    }
