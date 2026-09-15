"""Challenge 1: Generic Classifier Evaluator."""
from typing import Dict, Any
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

def evaluate_classifier(model, X_test, y_test) -> Dict[str, float]:
    """Calculate standard evaluation metrics for any trained classifier."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
        metrics['average_precision'] = float(average_precision_score(y_test, y_prob))
    return metrics

if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    res = evaluate_classifier(rf, X_te, y_te)
    print("Evaluation Results:", res)
    assert 'accuracy' in res and 'roc_auc' in res and res['roc_auc'] > 0.80
    print("Challenge 1 passed!")
