"""Experiment 1: Bag of Words feature representations."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.loader import load_sms_data
from app.data.cleaner import clean_dataset

def run_bow_experiment():
    print("--- Running Bag of Words Experiments ---")
    df = clean_dataset(load_sms_data(config.DATA_RAW))
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], np.where(df["label"] == "spam", 1, 0),
        test_size=0.25, random_state=42, stratify=np.where(df["label"] == "spam", 1, 0)
    )
    
    settings = [
        ("Binary BoW Unigrams", CountVectorizer(binary=True, ngram_range=(1, 1))),
        ("Count BoW Unigrams", CountVectorizer(binary=False, ngram_range=(1, 1))),
        ("Count BoW Uni+Bigrams", CountVectorizer(binary=False, ngram_range=(1, 2))),
    ]
    
    results = []
    for name, vec in settings:
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        results.append({"representation": name, "vocab_size": X_tr.shape[1], "accuracy": acc, "f1": f1})
        print(f"[{name}] Vocab: {X_tr.shape[1]} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    run_bow_experiment()
