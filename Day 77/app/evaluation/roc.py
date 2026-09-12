import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_binary_roc(y_true, y_prob) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val = float(roc_auc_score(y_true, y_prob))
    
    # Youden's J statistic = TPR - FPR
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc_val,
        "optimal_threshold_youden": best_threshold,
        "optimal_j": float(j_scores[best_idx])
    }

def compute_multiclass_roc(y_true, y_prob, classes=None) -> dict:
    if classes is None:
        classes = ["Low", "Medium", "High"]
        
    y_bin = label_binarize(y_true, classes=classes)
    res = {"classes": classes, "roc_curves": {}}
    
    for i, cls_name in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_prob[:, i])
        cls_auc = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        res["roc_curves"][cls_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": cls_auc
        }
        
    macro_auc = float(roc_auc_score(y_bin, y_prob, average="macro"))
    res["macro_auc"] = macro_auc
    return res
