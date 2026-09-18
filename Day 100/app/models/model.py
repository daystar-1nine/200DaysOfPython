
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    metrics = {
        'accuracy': round(accuracy_score(y_test, preds), 3),
        'precision': round(precision_score(y_test, preds, zero_division=0), 3),
        'recall': round(recall_score(y_test, preds, zero_division=0), 3),
        'f1': round(f1_score(y_test, preds, zero_division=0), 3)
    }
    return clf, metrics
