"""Challenge 3: Gini vs Entropy Cross-Validation."""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

def solve():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments"]
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for crit in ["gini", "entropy"]:
        clf = DecisionTreeClassifier(criterion=crit, max_depth=5, random_state=42)
        cv_scores = cross_validate(clf, X_train, y_train, cv=cv, scoring="f1")
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        print(f"=== {crit.upper()} ===")
        print(f"CV F1 (Mean):  {np.mean(cv_scores['test_score']):.4f}")
        print(f"Test F1:       {f1_score(y_test, preds):.4f}")
        print(f"Test Recall:   {recall_score(y_test, preds):.4f}")
        print(f"Test Precision:{precision_score(y_test, preds):.4f}")
        print(f"Test ROC-AUC:  {roc_auc_score(y_test, probs):.4f}\n")
        
    print("Conclusion: Gini and Entropy yield remarkably similar decision boundaries, with tiny differences stemming from logarithmic weighting.")

if __name__ == "__main__":
    solve()
