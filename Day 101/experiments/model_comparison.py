"""Experiment 4: Hyperparameter tuning and model comparison."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.loader import load_sms_data
from app.data.cleaner import clean_dataset

def run_model_comparison():
    print("--- Running Model Hyperparameter Experiments ---")
    df = clean_dataset(load_sms_data(config.DATA_RAW))
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], np.where(df["label"] == "spam", 1, 0),
        test_size=0.25, random_state=42, stratify=np.where(df["label"] == "spam", 1, 0)
    )
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    
    models = [
        ("Logistic Regression (C=0.1)", LogisticRegression(C=0.1, random_state=42)),
        ("Logistic Regression (C=1.0)", LogisticRegression(C=1.0, random_state=42)),
        ("Logistic Regression (C=10.0)", LogisticRegression(C=10.0, random_state=42)),
        ("Multinomial NB (alpha=0.1)", MultinomialNB(alpha=0.1)),
        ("Multinomial NB (alpha=1.0)", MultinomialNB(alpha=1.0)),
    ]
    
    records = []
    for name, clf in models:
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        records.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    run_model_comparison()
