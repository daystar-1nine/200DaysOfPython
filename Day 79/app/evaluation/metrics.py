from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    log_loss, brier_score_loss, confusion_matrix
)

def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate comprehensive set of classification performance metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(specificity),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
        metrics['log_loss'] = float(log_loss(y_true, y_prob))
        metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
        
    return metrics
