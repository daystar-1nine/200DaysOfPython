"""Educational from-scratch implementation of TF-IDF."""
import numpy as np
from typing import List, Dict, Tuple
from app.preprocessing.tokenizer import word_tokenize

class TFIDFScratch:
    """
    From-scratch implementation of TF-IDF matching scikit-learn's standard smoothed formulation:
      TF(t, d) = count(t, d) / total_words(d)
      IDF(t)   = log((1 + N) / (1 + df(t))) + 1
      TFIDF    = TF * IDF
      Normalized by row L2 norm: v / ||v||_2
    """
    def __init__(self, min_df: int = 1, sublinear_tf: bool = False):
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        self.n_docs_: int = 0

    def fit(self, documents: List[str]) -> "TFIDFScratch":
        tokenized_docs = [word_tokenize(doc) for doc in documents]
        self.n_docs_ = len(documents)
        
        # 1. Compute document frequency df(t)
        doc_counts: Dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_counts[token] = doc_counts.get(token, 0) + 1
                
        # 2. Filter vocabulary by min_df
        filtered_words = sorted([w for w, c in doc_counts.items() if c >= self.min_df])
        self.vocabulary_ = {word: idx for idx, word in enumerate(filtered_words)}
        vocab_size = len(self.vocabulary_)
        
        # 3. Compute smoothed IDF: log((1 + N) / (1 + df)) + 1
        df_arr = np.zeros(vocab_size, dtype=np.float64)
        for word, idx in self.vocabulary_.items():
            df_arr[idx] = doc_counts[word]
            
        self.idf_ = np.log((1.0 + self.n_docs_) / (1.0 + df_arr)) + 1.0
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise ValueError("TFIDFScratch must be fitted before calling transform.")
            
        n_docs = len(documents)
        vocab_size = len(self.vocabulary_)
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)
        
        for doc_idx, doc in enumerate(documents):
            tokens = word_tokenize(doc)
            total_tokens = len(tokens)
            if total_tokens == 0:
                continue
                
            counts: Dict[int, int] = {}
            for t in tokens:
                if t in self.vocabulary_:
                    idx = self.vocabulary_[t]
                    counts[idx] = counts.get(idx, 0) + 1
                    
            for idx, count in counts.items():
                if self.sublinear_tf:
                    tf = 1.0 + np.log(count)
                else:
                    tf = count / total_tokens
                matrix[doc_idx, idx] = tf * self.idf_[idx]
                
        # L2 Normalization
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> List[str]:
        inv = {idx: word for word, idx in self.vocabulary_.items()}
        return [inv[i] for i in range(len(inv))]
