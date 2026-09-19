"""Feature representation experiments."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from app.features.tfidf import get_word_tfidf, get_char_tfidf
from app.evaluation.cross_validation import run_stratified_cv

def run_feature_experiments(X, y) -> pd.DataFrame:
    """Compares 6 distinct feature engineering settings using 5-fold CV."""
    experiments = [
        ("Word Unigram", get_word_tfidf(ngram_range=(1, 1))),
        ("Word Uni+Bigram", get_word_tfidf(ngram_range=(1, 2))),
        ("Word Uni+Bi+Trigram", get_word_tfidf(ngram_range=(1, 3))),
        ("Character 3-5 Gram", get_char_tfidf(ngram_range=(3, 5))),
        ("Word Uni+Bi (min_df=1)", get_word_tfidf(ngram_range=(1, 2), min_df=1)),
        ("Word Uni+Bi (min_df=5)", get_word_tfidf(ngram_range=(1, 2), min_df=5)),
    ]
    
    results = []
    for name, vec in experiments:
        pipe = Pipeline([
            ("tfidf", vec),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", random_state=42))
        ])
        mean_f1, std_f1 = run_stratified_cv(pipe, X, y, n_splits=5)
        
        # Fit once on full training set to measure vocab size
        pipe.fit(X, y)
        vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
        
        results.append({
            "experiment": name,
            "vocab_size": vocab_size,
            "cv_f1_mean": mean_f1,
            "cv_f1_std": std_f1
        })
    return pd.DataFrame(results)
