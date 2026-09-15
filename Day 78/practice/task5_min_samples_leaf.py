"""Task 5: Minimum Samples per Leaf Regularization."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def run_task():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Age", "Tenure_Months", "Monthly_Charges", "Support_Calls", "Late_Payments"]
    X = df[features]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"{'min_samples_leaf':>16} | {'Leaf Count':>10} | {'Train F1':>10} | {'Test F1':>10}")
    print("-" * 54)
    for leaf in [1, 2, 5, 10, 20, 50]:
        clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train, y_train)
        n_leaves = clf.get_n_leaves()
        train_f1 = f1_score(y_train, clf.predict(X_train))
        test_f1 = f1_score(y_test, clf.predict(X_test))
        print(f"{leaf:>16} | {n_leaves:>10} | {train_f1:>10.4f} | {test_f1:>10.4f}")

if __name__ == "__main__":
    run_task()
