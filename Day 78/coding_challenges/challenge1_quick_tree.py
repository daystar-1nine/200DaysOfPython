"""Challenge 1: Build a Tree from Scratch."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def solve():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls"]
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print("=== Challenge 1: Independent Decision Tree Classifier ===")
    print(classification_report(y_test, preds, target_names=["Retained", "Churned"]))

if __name__ == "__main__":
    solve()
