import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

def calculate_binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    return metrics

def calculate_multiclass_metrics(y_true, y_pred, y_prob=None, labels=None) -> dict:
    if labels is None:
        labels = ["Low", "Medium", "High"]
        
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
    }
    
    # Per-class metrics
    prec_per = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": float(prec_per[idx]),
            "recall": float(rec_per[idx]),
            "f1": float(f1_per[idx])
        }
    metrics["per_class"] = per_class
    
    if y_prob is not None:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc_ovr"] = None
            
    return metrics

def compare_models(models_dict: dict, X_test, y_test, is_multiclass: bool = False) -> pd.DataFrame:
    rows = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None
            
        if not is_multiclass:
            prob_pos = y_prob[:, 1] if y_prob is not None else None
            m = calculate_binary_metrics(y_test, y_pred, prob_pos)
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "ROC-AUC": m.get("roc_auc", np.nan),
                "PR-AUC (AP)": m.get("average_precision", np.nan)
            })
        else:
            m = calculate_multiclass_metrics(y_test, y_pred, y_prob)
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Macro F1": m["macro_f1"],
                "Weighted F1": m["weighted_f1"],
                "ROC-AUC (OvR Macro)": m.get("roc_auc_ovr", np.nan),
                "Low Recall": m["per_class"].get("Low", {}).get("recall", np.nan),
                "Medium Recall": m["per_class"].get("Medium", {}).get("recall", np.nan),
                "High Recall": m["per_class"].get("High", {}).get("recall", np.nan)
            })
    return pd.DataFrame(rows)
