"""
Text embedders: Dense TF-IDF + SVD semantic representations, sparse TF-IDF, and word-centroid embedders.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class BaseEmbedder(ABC):
    """Abstract base class for all text embedding models."""
    
    @abstractmethod
    def fit(self, texts: List[str]) -> "BaseEmbedder":
        """Fits vocabulary and projection matrix on reference corpus."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Converts a single text string into a 1D vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Converts a list of text strings into a 2D numpy matrix of shape (N, D)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass


class DenseTFIDFEmbedder(BaseEmbedder):
    """
    Semantic dense embedder combining TF-IDF with Truncated SVD dimensionality reduction.
    Produces compact, normalized dense vector representations.
    """
    def __init__(self, dim: int = 64, ngram_range=(1, 2), max_features: int = 5000):
        self._dim = dim
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words="english"
        )
        self.svd: Optional[TruncatedSVD] = None
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        return self._dim

    def fit(self, texts: List[str]) -> "DenseTFIDFEmbedder":
        if not texts:
            raise ValueError("Cannot fit embedder on empty text collection.")
        
        # Fit TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_features = tfidf_matrix.shape[1]
        n_samples = tfidf_matrix.shape[0]

        # Determine safe SVD components
        target_dim = min(self._dim, n_features, n_samples)
        if target_dim > 1 and n_features > target_dim:
            self.svd = TruncatedSVD(n_components=target_dim, random_state=42)
            self.svd.fit(tfidf_matrix)
        else:
            self.svd = None

        self._is_fitted = True
        return self

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not self._is_fitted:
            # Auto-fit if not explicitly fitted
            self.fit(texts)

        if not texts:
            return np.empty((0, self._dim), dtype=np.float64)

        sparse_mat = self.vectorizer.transform(texts)
        if self.svd is not None:
            dense_mat = self.svd.transform(sparse_mat)
        else:
            dense_mat = sparse_mat.toarray()

        # Ensure exact target dimension
        n_rows, current_dim = dense_mat.shape
        if current_dim < self._dim:
            padded = np.zeros((n_rows, self._dim), dtype=np.float64)
            padded[:, :current_dim] = dense_mat
            dense_mat = padded
        elif current_dim > self._dim:
            dense_mat = dense_mat[:, :self._dim]

        # L2 row normalization
        norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = dense_mat / norms
        return normalized.astype(np.float64)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


class SparseTFIDFEmbedder(BaseEmbedder):
    """
    High-dimensional sparse TF-IDF embedder.
    """
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words="english"
        )
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        if self._is_fitted:
            return len(self.vectorizer.vocabulary_)
        return self.max_features

    def fit(self, texts: List[str]) -> "SparseTFIDFEmbedder":
        if not texts:
            raise ValueError("Cannot fit on empty texts.")
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not self._is_fitted:
            self.fit(texts)
        sparse_mat = self.vectorizer.transform(texts)
        arr = sparse_mat.toarray().astype(np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return arr / norms

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


def get_embedder(name: str = "dense_tfidf", dim: int = 64) -> BaseEmbedder:
    """Factory function for instantiating embedders."""
    name_clean = name.lower().strip()
    if name_clean in {"dense", "dense_tfidf", "svd"}:
        return DenseTFIDFEmbedder(dim=dim)
    elif name_clean in {"sparse", "sparse_tfidf", "tfidf"}:
        return SparseTFIDFEmbedder()
    else:
        raise ValueError(f"Unknown embedder type '{name}'. Choose from 'dense_tfidf', 'sparse_tfidf'.")
