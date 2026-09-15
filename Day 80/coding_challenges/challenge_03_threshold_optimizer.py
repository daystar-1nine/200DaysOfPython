"""Challenge 3: Financial Threshold Optimizer."""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray, fp_cost: float = 300.0, fn_cost: float = 2000.0) -> Dict[str, float]:
    """Find decision threshold that minimizes business misclassification cost."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    min_cost = float('inf')
    best_prec, best_rec, best_f1 = 0.0, 0.0, 0.0
    
    for t in thresholds:
        y_pred = (probabilities >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < min_cost:
            min_cost = cost
            best_t = float(t)
            best_prec = float(precision_score(y_true, y_pred, zero_division=0))
            best_rec = float(recall_score(y_true, y_pred, zero_division=0))
            best_f1 = float(f1_score(y_true, y_pred, zero_division=0))
            
    return {
        'best_threshold': best_t,
        'minimum_cost': min_cost,
        'precision': best_prec,
        'recall': best_rec,
        'f1': best_f1
    }

if __name__ == '__main__':
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0] * 10)
    probs = np.array([0.8, 0.2, 0.7, 0.4, 0.1, 0.3, 0.9, 0.6, 0.75, 0.15] * 10)
    res = find_best_threshold(y_true, probs, fp_cost=300.0, fn_cost=2000.0)
    print("Optimal Threshold Results:", res)
    assert 0.0 < res['best_threshold'] < 1.0
    assert res['minimum_cost'] < (len(y_true) * 2000.0)
    print("Challenge 3 passed!")
