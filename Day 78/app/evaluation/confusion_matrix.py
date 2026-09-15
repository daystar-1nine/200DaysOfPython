import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_true, y_pred, labels=None) -> dict:
    if labels is None:
        labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    cm_df = pd.DataFrame(cm, index=["Actual Retained (0)", "Actual Churned (1)"],
                         columns=["Predicted Retained (0)", "Predicted Churned (1)"])
    return {
        "matrix": cm,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "dataframe": cm_df
    }
