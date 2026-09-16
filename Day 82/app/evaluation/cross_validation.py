
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_cv(model, X, y, model_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=['roc_auc', 'average_precision'])
    
    return {
        'Model': model_name,
        'Mean CV ROC-AUC': np.mean(scores['test_roc_auc']),
        'Std CV ROC-AUC': np.std(scores['test_roc_auc']),
        'Mean Average Precision': np.mean(scores['test_average_precision']),
        'Std Average Precision': np.std(scores['test_average_precision'])
    }
