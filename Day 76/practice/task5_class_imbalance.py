# Task 5: Class Imbalance
# Train a logistic regression model with class weights

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def run_imbalance():
    X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Without class weights:")
    model1 = LogisticRegression()
    model1.fit(X_train, y_train)
    print(classification_report(y_test, model1.predict(X_test)))
    
    print("\nWith balanced class weights:")
    model2 = LogisticRegression(class_weight='balanced')
    model2.fit(X_train, y_train)
    print(classification_report(y_test, model2.predict(X_test)))

if __name__ == "__main__":
    run_imbalance()
