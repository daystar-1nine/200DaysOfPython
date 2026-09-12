import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def compute_binary_pr(y_true, y_prob) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": ap
    }

def compute_multiclass_pr(y_true, y_prob, classes=None) -> dict:
    if classes is None:
        classes = ["Low", "Medium", "High"]
        
    y_bin = label_binarize(y_true, classes=classes)
    res = {"classes": classes, "pr_curves": {}}
    
    for i, cls_name in enumerate(classes):
        p, r, thresh = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = float(average_precision_score(y_bin[:, i], y_prob[:, i]))
        res["pr_curves"][cls_name] = {
            "precision": p,
            "recall": r,
            "thresholds": thresh,
            "average_precision": ap
        }
    res["macro_ap"] = float(np.mean([res["pr_curves"][c]["average_precision"] for c in classes]))
    return res
