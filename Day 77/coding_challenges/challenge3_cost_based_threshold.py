"""Challenge 3: Cost-Based Threshold Optimization."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def solve():
    X, y = make_classification(n_samples=1000, n_features=6, weights=[0.80, 0.20], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    fp_cost = 300.0
    fn_cost = 5000.0
    
    best_t, min_cost = 0.50, float("inf")
    cost_at_50 = 0.0
    
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cost = fp * fp_cost + fn * fn_cost
        if round(t, 2) == 0.50:
            cost_at_50 = cost
        if cost < min_cost:
            min_cost = cost
            best_t = t
            
    print(f"Cost-Optimal Threshold: tau = {best_t:.2f}")
    print(f"Minimal Business Cost:   INR {min_cost:,.2f}")
    print(f"Default Cost (tau=0.50): INR {cost_at_50:,.2f}")
    print(f"Projected Savings:       INR {cost_at_50 - min_cost:,.2f}")

if __name__ == "__main__":
    solve()
