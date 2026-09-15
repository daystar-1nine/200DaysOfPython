"""Task 3: Tree Depth vs Overfitting Experiment."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def run_task():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments", "Complaints"]
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    depths = [1, 2, 3, 4, 5, 7, 10, 15, None]
    print(f"{'Max Depth':>10} | {'Train F1':>10} | {'Test F1':>10} | {'Status':<15}")
    print("-" * 52)
    
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        
        train_f1 = f1_score(y_train, clf.predict(X_train))
        test_f1 = f1_score(y_test, clf.predict(X_test))
        
        gap = train_f1 - test_f1
        status = "Good Fit" if gap < 0.05 else ("Moderate Overfit" if gap < 0.15 else "Severe Overfit")
        d_str = str(d) if d is not None else "None"
        print(f"{d_str:>10} | {train_f1:>10.4f} | {test_f1:>10.4f} | {status:<15}")

if __name__ == "__main__":
    run_task()
