"""Challenge 2: Cross-Validation Model Comparison."""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def compare_models(models: Dict[str, Any], X, y, cv=5, scoring='roc_auc') -> pd.DataFrame:
    """Compare multiple models using cross-validation and return summary DataFrame."""
    records = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        records.append({
            'model': name,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores))
        })
    df = pd.DataFrame(records)
    return df.sort_values(by='mean_score', ascending=False).reset_index(drop=True)

if __name__ == '__main__':
    X, y = make_classification(n_samples=600, n_features=10, random_state=42)
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    df_comp = compare_models(candidates, X, y, cv=5)
    print(df_comp)
    assert len(df_comp) == 3
    assert df_comp.loc[0, 'mean_score'] > 0.85
    print("Challenge 2 passed!")
