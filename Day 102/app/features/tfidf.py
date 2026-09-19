"""Configured scikit-learn TF-IDF vectorizer builders."""
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional

def get_word_tfidf(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 1.0,
    max_features: Optional[int] = 10000,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Builds configured word-level TfidfVectorizer."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        stop_words="english"
    )

def get_char_tfidf(
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: Optional[int] = 10000,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Builds configured character-level TfidfVectorizer."""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=True
    )
