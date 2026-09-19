"""Experiment comparing Word vs Character N-grams."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.loader import load_data
from app.features.tfidf import get_word_tfidf, get_char_tfidf

def run():
    print("=== EXPERIMENT: WORD VS CHARACTER N-GRAMS ===")
    df = load_data(config.DATA_RAW)
    X = df["text"].values
    y = np.where(df["label"] == "spam", 1, 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    pipes = [
        ("Word TF-IDF (1, 2)", get_word_tfidf(ngram_range=(1, 2))),
        ("Char TF-IDF (3, 5)", get_char_tfidf(ngram_range=(3, 5))),
    ]
    
    for name, vec in pipes:
        p = Pipeline([("tfidf", vec), ("clf", LogisticRegression(class_weight="balanced", random_state=42))])
        p.fit(X_train, y_train)
        preds = p.predict(X_test)
        vocab_size = len(p.named_steps["tfidf"].vocabulary_)
        print(f"\n--- {name} (Vocab Size: {vocab_size}) ---")
        print(classification_report(y_test, preds, target_names=["Ham", "Spam"], digits=4))

if __name__ == "__main__":
    run()
