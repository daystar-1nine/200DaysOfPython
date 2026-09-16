
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_cv(model, X, y, model_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=['roc_auc', 'average_precision'], n_jobs=-1)
    
    return {
        'Model': model_name,
        'CV ROC-AUC': np.mean(scores['test_roc_auc']),
        'CV Std ROC': np.std(scores['test_roc_auc']),
        'CV AP': np.mean(scores['test_average_precision']),
        'CV Std AP': np.std(scores['test_average_precision'])
    }
