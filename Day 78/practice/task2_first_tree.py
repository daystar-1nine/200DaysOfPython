"""Task 2: Train First Decision Tree Classifier."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_task():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments"]
    
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    print("=== First Decision Tree Performance (Depth=5) ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1-Score:  {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
    run_task()
