"""Task 6: Feature Importance Extraction."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def run_task():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments", "Complaints"]
    X = df[features]
    y = df["Churn"]
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    print("=== Decision Tree Feature Importances (MDI) ===")
    print(imp_df.to_string(index=False))

if __name__ == "__main__":
    run_task()
