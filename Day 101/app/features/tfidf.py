"""TF-IDF feature extraction module."""
import numpy as np
import math
from typing import List, Dict, Tuple
from app.preprocessing.tokenization import tokenize
from app.preprocessing.vocabulary import build_vocabulary
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFScratch:
    """
    Educational from-scratch implementation of TF-IDF.
    Formula:
      TF(t, d) = count(t, d) / total_words(d)
      IDF(t)   = log(N / (df(t) + 1)) + 1
      TFIDF    = TF * IDF
    """
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        self.n_docs_: int = 0

    def fit(self, documents: List[str]) -> "TFIDFScratch":
        tokenized_docs = [tokenize(doc) for doc in documents]
        self.vocabulary_ = build_vocabulary(tokenized_docs, min_freq=self.min_freq)
        self.n_docs_ = len(documents)
        vocab_size = len(self.vocabulary_)
        
        # Calculate document frequency df(t)
        df = np.zeros(vocab_size, dtype=np.float64)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary_:
                    df[self.vocabulary_[token]] += 1
                    
        # Smoothed IDF: log((1 + N) / (1 + df)) + 1
        self.idf_ = np.log((1.0 + self.n_docs_) / (1.0 + df)) + 1.0
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise ValueError("TFIDFScratch instance is not fitted yet.")
            
        n_docs = len(documents)
        vocab_size = len(self.vocabulary_)
        tfidf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)
        
        for doc_idx, doc in enumerate(documents):
            tokens = tokenize(doc)
            total_tokens = len(tokens)
            if total_tokens == 0:
                continue
                
            # Count terms
            counts: Dict[int, int] = {}
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    counts[idx] = counts.get(idx, 0) + 1
                    
            for idx, count in counts.items():
                tf = count / total_tokens
                tfidf_matrix[doc_idx, idx] = tf * self.idf_[idx]
                
        # L2 Normalization per row
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return tfidf_matrix / norms

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

def get_sklearn_tfidf(ngram_range: Tuple[int, int] = (1, 1), min_df: int = 2, max_features: Optional[int] = 1000) -> TfidfVectorizer:
    """Returns configured scikit-learn TfidfVectorizer for pipeline integration."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        stop_words="english",
        lowercase=True
    )
