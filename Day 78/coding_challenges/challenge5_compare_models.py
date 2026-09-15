"""Challenge 5: Model Selection and Architecture Tradeoffs."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score

def solve():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments", "Complaints"]
    X = df[features]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    log = LogisticRegression(random_state=42).fit(X_train_s, y_train)
    tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
    
    p_log, prob_log = log.predict(X_test_s), log.predict_proba(X_test_s)[:, 1]
    p_tree, prob_tree = tree.predict(X_test), tree.predict_proba(X_test)[:, 1]
    
    print("=== Model Architecture Evaluation ===")
    print(f"Metric        | Logistic Regression | Decision Tree (Depth=5)")
    print("-" * 55)
    print(f"F1-Score      | {f1_score(y_test, p_log):>19.4f} | {f1_score(y_test, p_tree):>23.4f}")
    print(f"Recall        | {recall_score(y_test, p_log):>19.4f} | {recall_score(y_test, p_tree):>23.4f}")
    print(f"ROC-AUC       | {roc_auc_score(y_test, prob_log):>19.4f} | {roc_auc_score(y_test, prob_tree):>23.4f}")
    print(f"Interpret     | Linear coefficients | Human-readable IF-THEN rules")
    print(f"Scaling Req?  | Yes (Z-score)       | No (Monotonic invariance)")

if __name__ == "__main__":
    solve()
