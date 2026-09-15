"""Challenge 2: Find Optimal Tree Depth via 5-Fold Stratified CV."""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def solve():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments", "Complaints"]
    X = df[features]
    y = df["Churn"]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_depth = 1
    best_score = -1.0
    
    print("Sweeping depths 1 to 15:")
    for d in range(1, 16):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
        mean_f1 = np.mean(scores)
        print(f"Depth {d:2d} -> CV Mean F1: {mean_f1:.4f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_depth = d
            
    print(f"\nOptimal Tree Depth: {best_depth} (Best CV F1 = {best_score:.4f})")

if __name__ == "__main__":
    solve()
