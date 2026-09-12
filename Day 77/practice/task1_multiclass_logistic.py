"""Task 1: Multi-Class Logistic Regression Demonstration."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_task():
    # 3 classes: Low (0), Medium (1), High (2)
    X, y = make_classification(
        n_samples=500, n_features=6, n_informative=4, n_classes=3,
        weights=[0.60, 0.25, 0.15], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    print("Multi-Class Logistic Regression Sample Probabilities (First 5):")
    target_names = ["Low (0)", "Medium (1)", "High (2)"]
    for i in range(5):
        print(f"Sample {i+1}: Low={probs[i, 0]:.3f}, Medium={probs[i, 1]:.3f}, High={probs[i, 2]:.3f} -> Predicted: {target_names[preds[i]]}")
        
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=target_names))

if __name__ == "__main__":
    run_task()
