"""Task 4: Precision-Recall Curve vs ROC Analysis."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def run_task():
    X, y = make_classification(n_samples=1000, n_features=6, weights=[0.95, 0.05], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    
    print(f"ROC-AUC:            {roc_auc:.4f} (Can look deceptively high due to huge true negative pool)")
    print(f"Average Precision:  {ap:.4f} (Directly focuses on precision-recall for rare positive class)")

if __name__ == "__main__":
    run_task()
