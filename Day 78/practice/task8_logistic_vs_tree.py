"""Task 8: Logistic Regression vs Decision Tree Direct Comparison."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_task():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments"]
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
    
    models = {
        "Logistic Regression": (log_reg.predict(X_test_scaled), log_reg.predict_proba(X_test_scaled)[:, 1]),
        "Decision Tree (Depth=5)": (tree_clf.predict(X_test), tree_clf.predict_proba(X_test)[:, 1])
    }
    
    print(f"{'Model':<25} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'ROC-AUC':>8}")
    print("-" * 75)
    for name, (pred, prob) in models.items():
        print(f"{name:<25} | {accuracy_score(y_test, pred):>8.4f} | {precision_score(y_test, pred):>9.4f} | "
              f"{recall_score(y_test, pred):>8.4f} | {f1_score(y_test, pred):>8.4f} | {roc_auc_score(y_test, prob):>8.4f}")

if __name__ == "__main__":
    run_task()
