"""Task 3: Unweighted vs Balanced Logistic Regression Comparison."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

def run_task():
    X, y = make_classification(n_samples=1000, n_features=6, weights=[0.92, 0.08], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Unweighted
    model_unweighted = LogisticRegression(max_iter=1000, random_state=42)
    model_unweighted.fit(X_train, y_train)
    preds_unw = model_unweighted.predict(X_test)
    
    # Balanced
    model_balanced = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model_balanced.fit(X_train, y_train)
    preds_bal = model_balanced.predict(X_test)
    
    print("=== MODEL A: class_weight=None ===")
    print(f"Minority Recall:    {recall_score(y_test, preds_unw):.4f}")
    print(f"Minority Precision: {precision_score(y_test, preds_unw, zero_division=0):.4f}")
    print(f"Macro F1:           {f1_score(y_test, preds_unw, average='macro'):.4f}")
    
    print("\n=== MODEL B: class_weight='balanced' ===")
    print(f"Minority Recall:    {recall_score(y_test, preds_bal):.4f}")
    print(f"Minority Precision: {precision_score(y_test, preds_bal, zero_division=0):.4f}")
    print(f"Macro F1:           {f1_score(y_test, preds_bal, average='macro'):.4f}")

if __name__ == "__main__":
    run_task()
