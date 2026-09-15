"""Task 7: Tree Text Structure and Rules."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

def run_task():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Tenure_Months", "Support_Calls", "Monthly_Charges"]
    X = df[features]
    y = df["Churn"]
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    rules = export_text(clf, feature_names=features, class_names=["Stay", "Churn"])
    print("=== Learned Decision Tree Structure (Depth=3) ===")
    print(rules)

if __name__ == "__main__":
    run_task()
