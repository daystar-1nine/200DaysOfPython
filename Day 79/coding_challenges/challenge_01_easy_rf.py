"""Challenge 1: Train Basic Random Forest and Print Accuracy, Precision, Recall, F1."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def run_challenge_01():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    rep = classification_report(y_test, y_pred, output_dict=True)
    print(f"Accuracy: {rep['accuracy']:.4f} | F1-Score: {rep['weighted avg']['f1-score']:.4f}")
    return rep['accuracy']

if __name__ == '__main__':
    acc = run_challenge_01()
    assert acc > 0.85
    print("Challenge 1 passed!")
