import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

def run_stratified_cv(pipeline, X, y, n_splits: int = 5, random_state: int = 42, is_multiclass: bool = False) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted"
    }
    if not is_multiclass:
        scoring["roc_auc"] = "roc_auc"
        scoring["precision"] = "precision"
        scoring["recall"] = "recall"
    else:
        scoring["roc_auc_ovr"] = "roc_auc_ovr"
        
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    
    results = {}
    for metric, values in scores.items():
        clean_metric = metric[5:] if metric.startswith("test_") else metric
        results[f"mean_{clean_metric}"] = float(np.mean(values))
        results[f"std_{clean_metric}"] = float(np.std(values))
        results[f"all_{clean_metric}"] = [float(v) for v in values]
    return results
