"""Bag of Words representation from scratch."""
import numpy as np
from typing import List, Dict, Optional
from app.preprocessing.tokenization import tokenize
from app.preprocessing.vocabulary import build_vocabulary

class BagOfWords:
    """
    From-scratch implementation of Bag of Words (Document-Term Matrix).
    Does NOT use scikit-learn CountVectorizer.
    """
    def __init__(self, min_freq: int = 1, binary: bool = False):
        self.min_freq = min_freq
        self.binary = binary
        self.vocabulary_: Dict[str, int] = {}
        self.inverse_vocab_: Dict[int, str] = {}

    def fit(self, documents: List[str]) -> "BagOfWords":
        """Builds vocabulary from raw document texts."""
        tokenized_docs = [tokenize(doc) for doc in documents]
        self.vocabulary_ = build_vocabulary(tokenized_docs, min_freq=self.min_freq)
        self.inverse_vocab_ = {idx: word for word, idx in self.vocabulary_.items()}
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transforms documents into a 2D Document-Term Matrix."""
        if not self.vocabulary_:
            raise ValueError("BagOfWords instance is not fitted yet. Call 'fit' first.")
            
        n_docs = len(documents)
        vocab_size = len(self.vocabulary_)
        dtm = np.zeros((n_docs, vocab_size), dtype=np.int32)
        
        for doc_idx, doc in enumerate(documents):
            tokens = tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    word_idx = self.vocabulary_[token]
                    if self.binary:
                        dtm[doc_idx, word_idx] = 1
                    else:
                        dtm[doc_idx, word_idx] += 1
                        
        return dtm

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fits vocabulary and returns transformed Document-Term Matrix."""
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> List[str]:
        """Returns sorted list of vocabulary terms."""
        return [self.inverse_vocab_[i] for i in range(len(self.inverse_vocab_))]
