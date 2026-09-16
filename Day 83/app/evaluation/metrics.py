
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'Test ROC-AUC': roc_auc_score(y_test, y_prob) if any(y_prob) else 0.5,
        'Average Precision': average_precision_score(y_test, y_prob) if any(y_prob) else 0.5,
        'Train Time': training_time
    }
