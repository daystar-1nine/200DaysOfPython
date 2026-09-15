from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_stratified_cv(
    pipeline, 
    X: pd.DataFrame, 
    y: pd.Series, 
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """Perform Stratified 5-Fold Cross Validation across multiple classification metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
    
    summary = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        summary[metric] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
    return summary
