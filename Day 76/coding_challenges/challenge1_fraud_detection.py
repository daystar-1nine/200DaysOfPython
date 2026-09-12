# Challenge 1: Fraud Detection
# Build a simple logistic regression pipeline for fraud detection with synthetic data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def detect_fraud():
    X, y = make_classification(n_samples=5000, n_features=10, weights=[0.95, 0.05], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    f1 = f1_score(y_test, preds)
    
    print(f"Fraud Detection F1 Score: {f1:.2f}")

if __name__ == "__main__":
    detect_fraud()
