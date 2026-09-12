"""Challenge 1: Minority Class Champion."""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def solve():
    # Highly imbalanced dataset: 95% Class 0, 5% Class 1
    X, y = make_classification(n_samples=1500, n_features=8, weights=[0.95, 0.05], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    mA = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    mB = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(X_train, y_train)
    
    pA, pB = mA.predict(X_test), mB.predict(X_test)
    
    print("=== MODEL A (Unweighted) ===")
    print(f"Accuracy:          {accuracy_score(y_test, pA):.4f}")
    print(f"Minority Recall:   {recall_score(y_test, pA):.4f}")
    print(f"Minority F1:       {f1_score(y_test, pA):.4f}")
    print(f"Macro F1:          {f1_score(y_test, pA, average='macro'):.4f}")
    
    print("\n=== MODEL B (Balanced) ===")
    print(f"Accuracy:          {accuracy_score(y_test, pB):.4f}")
    print(f"Minority Recall:   {recall_score(y_test, pB):.4f}")
    print(f"Minority F1:       {f1_score(y_test, pB):.4f}")
    print(f"Macro F1:          {f1_score(y_test, pB, average='macro'):.4f}")
    
    print("\nDeployment Decision:")
    print("Deploy Model B (Balanced): Despite lower overall accuracy, it successfully captures the critical minority class.")

if __name__ == "__main__":
    solve()
