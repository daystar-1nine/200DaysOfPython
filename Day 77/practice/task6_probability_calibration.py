"""Task 6: Probability Calibration and Brier Score."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def run_task():
    X, y = make_classification(n_samples=1000, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=5)
    brier = brier_score_loss(y_test, probs)
    
    print("Probability Calibration Bin Results:")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"Mean Predicted Prob: {pp:.3f} -> True Positive Proportion: {pt:.3f}")
        
    print(f"\nOverall Brier Score (Lower is better, 0 = perfect): {brier:.4f}")

if __name__ == "__main__":
    run_task()
