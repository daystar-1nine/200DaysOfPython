import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred) -> float:
    return precision_score(y_true, y_pred, zero_division=0)

def calculate_recall(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, zero_division=0)

def calculate_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, zero_division=0)

def calculate_roc_auc(y_true, probabilities) -> float:
    return roc_auc_score(y_true, probabilities)

def calculate_all_metrics(y_true, y_pred, probabilities=None) -> dict:
    metrics = {
        'Accuracy': calculate_accuracy(y_true, y_pred),
        'Precision': calculate_precision(y_true, y_pred),
        'Recall': calculate_recall(y_true, y_pred),
        'F1': calculate_f1(y_true, y_pred)
    }
    
    if probabilities is not None:
        metrics['ROC_AUC'] = calculate_roc_auc(y_true, probabilities)
        
    return metrics
