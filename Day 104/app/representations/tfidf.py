"""
TF-IDF representation module for Day 104 Semantic Search Engine.
Fits TF-IDF on documents and transforms queries into the identical vector space.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRepresentation:
    """Manages TF-IDF vectorization for documents and search queries."""

    def __init__(
        self,
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
        norm: str = "l2"
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            lowercase=True
        )
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, texts: List[str]) -> "TfidfRepresentation":
        """Fit the TF-IDF vectorizer on a collection of document texts.
        
        Args:
            texts: List of document text strings.
            
        Returns:
            self
        """
        if not texts:
            raise ValueError("Cannot fit TF-IDF on empty text list.")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        return self

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Transform texts into TF-IDF vectors using the fitted vocabulary.
        
        Args:
            texts: Single text or list of texts.
            
        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        if not self.is_fitted:
            raise RuntimeError("TfidfRepresentation must be fitted before calling transform.")

        if isinstance(texts, str):
            texts = [texts]

        sparse_matrix = self.vectorizer.transform(texts)
        return sparse_matrix.toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer on texts and transform them into TF-IDF vectors."""
        return self.fit(texts).transform(texts)

    def get_token_weights(self, text: str) -> Dict[str, float]:
        """Extract non-zero TF-IDF weights for tokens present in a text.
        
        Useful for TF-IDF weighted pooling in document embeddings.
        
        Args:
            text: Input text string.
            
        Returns:
            Dictionary mapping word token to its TF-IDF weight.
        """
        if not self.is_fitted:
            raise RuntimeError("TfidfRepresentation must be fitted before extracting weights.")

        vec = self.transform(text)[0]
        nonzero_indices = np.nonzero(vec)[0]
        weights: Dict[str, float] = {}
        for idx in nonzero_indices:
            word = self.feature_names[idx]
            weights[word] = float(vec[idx])
        return weights

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.feature_names)
