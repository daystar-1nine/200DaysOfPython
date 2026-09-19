"""Experiment: Comparing Minimal vs Punctuation vs Aggressive Preprocessing."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.loader import load_data
from app.preprocessing.cleaner import clean_minimal, clean_punctuation, clean_aggressive

def run():
    print("=== EXPERIMENT: PREPROCESSING STRATEGIES ===")
    df = load_data(config.DATA_RAW)
    y = np.where(df["label"] == "spam", 1, 0)
    
    strategies = [
        ("Minimal (Lowercase & Whitespace)", df["text"].apply(clean_minimal).values),
        ("Punctuation Stripped", df["text"].apply(clean_punctuation).values),
        ("Aggressive (No URLs, numbers, short words)", df["text"].apply(clean_aggressive).values),
    ]
    
    for name, X in strategies:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42))
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1")
        print(f"[{name}] 5-Fold CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")

if __name__ == "__main__":
    run()
