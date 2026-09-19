"""Bag of Words helper."""
from sklearn.feature_extraction.text import CountVectorizer

def get_count_vectorizer(ngram_range=(1, 1), min_df=2, max_features=5000) -> CountVectorizer:
    return CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features, lowercase=True)
