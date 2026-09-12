"""Challenge 2: Recall-Constrained Optimal Threshold."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

def solve():
    X, y = make_classification(n_samples=1000, n_features=6, weights=[0.75, 0.25], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    candidates = []
    for t in np.linspace(0.05, 0.95, 37):
        preds = (probs >= t).astype(int)
        r = recall_score(y_test, preds, zero_division=0)
        p = precision_score(y_test, preds, zero_division=0)
        if r >= 0.85:
            candidates.append((t, p, r))
            
    # Sort candidates by precision descending
    best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    print(f"Optimal Threshold with Recall >= 85%: tau = {best[0]:.2f}")
    print(f"Achieved Recall:    {best[2]:.4f}")
    print(f"Achieved Precision: {best[1]:.4f}")

if __name__ == "__main__":
    solve()
