"""Challenge 3: Gini Impurity vs Entropy Criterion Comparison."""
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def compare_criteria():
    X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
    gini_rf = RandomForestClassifier(n_estimators=50, criterion='gini', random_state=42)
    entropy_rf = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
    
    gini_scores = cross_val_score(gini_rf, X, y, cv=5, scoring='f1')
    entropy_scores = cross_val_score(entropy_rf, X, y, cv=5, scoring='f1')
    
    print(f"Gini F1:    {np.mean(gini_scores):.4f} (+/- {np.std(gini_scores):.4f})")
    print(f"Entropy F1: {np.mean(entropy_scores):.4f} (+/- {np.std(entropy_scores):.4f})")
    return np.mean(gini_scores), np.mean(entropy_scores)

if __name__ == '__main__':
    g, e = compare_criteria()
    assert g > 0.80 and e > 0.80
    print("Challenge 3 passed!")
