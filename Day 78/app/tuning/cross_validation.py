import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

def run_stratified_cv(pipeline, X, y, n_splits: int = 5, random_state: int = 42) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    results = {}
    for k, v in scores.items():
        clean_name = k[5:] if k.startswith("test_") else (k[6:] + "_train" if k.startswith("train_") else k)
        results[f"mean_{clean_name}"] = float(np.mean(v))
        results[f"std_{clean_name}"] = float(np.std(v))
        results[f"all_{clean_name}"] = [float(x) for x in v]
    return results

def evaluate_depth_curve(preprocessor, X_train, y_train, X_test, y_test, depths=None, random_state=42) -> pd.DataFrame:
    from app.models.decision_tree import build_decision_tree_model
    from sklearn.metrics import accuracy_score, f1_score
    
    if depths is None:
        depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
        
    records = []
    for d in depths:
        m = build_decision_tree_model(preprocessor, max_depth=d, random_state=random_state)
        m.fit(X_train, y_train)
        
        train_pred = m.predict(X_train)
        test_pred = m.predict(X_test)
        
        d_label = str(d) if d is not None else "None"
        records.append({
            "Depth": d_label,
            "Numeric_Depth": d if d is not None else 20,
            "Train_Accuracy": float(accuracy_score(y_train, train_pred)),
            "Test_Accuracy": float(accuracy_score(y_test, test_pred)),
            "Train_F1": float(f1_score(y_train, train_pred)),
            "Test_F1": float(f1_score(y_test, test_pred))
        })
    return pd.DataFrame(records)
