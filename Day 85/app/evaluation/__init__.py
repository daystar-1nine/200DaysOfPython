
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_churn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'CM': confusion_matrix(y_test, y_pred)
    }

def evaluate_text(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Report': classification_report(y_test, y_pred, zero_division=0)
    }

def calculate_business_cost(y_true, y_prob, threshold, fp_cost=100, fn_cost=1000):
    y_pred = (y_prob >= threshold).astype(int)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return (fp * fp_cost) + (fn * fn_cost)
