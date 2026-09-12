# Task 1: Logistic Regression Demo
# Implement a basic Logistic Regression model using sklearn on a dummy dataset.

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def run_demo():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score:.2f}")

if __name__ == "__main__":
    run_demo()
