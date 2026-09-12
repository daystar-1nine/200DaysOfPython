import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "matrix": cm
    }

def format_confusion_matrix_summary(cm_dict: dict) -> str:
    summary = f"""Confusion Matrix Summary:
True Positives (TP): {cm_dict['TP']}
True Negatives (TN): {cm_dict['TN']}
False Positives (FP): {cm_dict['FP']}
False Negatives (FN): {cm_dict['FN']}
"""
    return summary
