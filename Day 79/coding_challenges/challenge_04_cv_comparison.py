"""Challenge 4: 5-Fold Stratified Cross-Validation: Decision Tree vs Random Forest."""
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def cv_tree_vs_forest():
    X, y = make_classification(n_samples=1200, n_features=12, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    dt_f1 = cross_val_score(dt, X, y, cv=skf, scoring='f1')
    rf_f1 = cross_val_score(rf, X, y, cv=skf, scoring='f1')
    
    print(f"Decision Tree F1: {np.mean(dt_f1):.4f} (+/- {np.std(dt_f1):.4f})")
    print(f"Random Forest F1: {np.mean(rf_f1):.4f} (+/- {np.std(rf_f1):.4f})")
    return np.mean(dt_f1), np.mean(rf_f1)

if __name__ == '__main__':
    dt_score, rf_score = cv_tree_vs_forest()
    assert rf_score >= dt_score
    print("Challenge 4 passed!")
