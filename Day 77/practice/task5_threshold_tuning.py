"""Task 5: Classification Threshold Sweeping & Optimization."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def run_task():
    X, y = make_classification(n_samples=800, n_features=6, weights=[0.80, 0.20], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("Threshold Sweep:")
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 46)
    
    best_f1, best_t = 0.0, 0.50
    for t in np.linspace(0.10, 0.90, 9):
        preds = (probs >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
        print(f"{t:>10.2f} | {p:>10.4f} | {r:>10.4f} | {f:>10.4f}")
        
    print(f"\nOptimal F1 Threshold: {best_t:.2f} (Max F1 = {best_f1:.4f})")

if __name__ == "__main__":
    run_task()
