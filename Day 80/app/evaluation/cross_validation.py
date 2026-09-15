from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

def run_stratified_cv(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """Evaluate pipeline using 5-fold Stratified Cross-Validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'
    }
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=True)
    
    summary = {}
    for metric in scoring.keys():
        test_vals = scores[f'test_{metric}']
        train_vals = scores[f'train_{metric}']
        summary[metric] = {
            'train_mean': float(np.mean(train_vals)),
            'test_mean': float(np.mean(test_vals)),
            'test_std': float(np.std(test_vals)),
            'test_min': float(np.min(test_vals)),
            'test_max': float(np.max(test_vals))
        }
    return summary
