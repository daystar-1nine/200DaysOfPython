"""Challenge 4: ROC-AUC vs PR-AUC on Highly Imbalanced Data."""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def solve():
    # Rare event: 99% Negative, 1% Positive
    X, y = make_classification(n_samples=2000, n_features=8, weights=[0.99, 0.01], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    
    print(f"ROC-AUC:           {roc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("\nExplanation:")
    print("ROC-AUC uses FPR (FP / Total Negatives). When negatives vastly outnumber positives, hundreds of false positives")
    print("yield only a tiny change in FPR, keeping ROC-AUC high. In contrast, Precision (TP / (TP + FP)) directly penalizes")
    print("false positives against the small true positive count, providing a truthful evaluation for rare positive events.")

if __name__ == "__main__":
    solve()
