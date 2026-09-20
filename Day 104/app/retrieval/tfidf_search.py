"""
TF-IDF Search Engine module for Day 104 Semantic Search Engine.
Implements keyword-based retrieval using TF-IDF and cosine similarity.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from ..preprocessing.cleaner import TextCleaner
from ..representations.tfidf import TfidfRepresentation
from .ranking import compute_similarities, rank_top_k


class TfidfSearchEngine:
    """Keyword search engine utilizing TF-IDF vectors and cosine similarity ranking."""

    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True
    ):
        self.cleaner = cleaner if cleaner is not None else TextCleaner()
        self.tfidf = TfidfRepresentation(ngram_range=ngram_range, sublinear_tf=sublinear_tf)
        self.documents: List[Dict[str, Any]] = []
        self.doc_matrix: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def index(self, documents: List[Dict[str, Any]]) -> "TfidfSearchEngine":
        """Index a collection of documents by fitting TF-IDF vectors.
        
        Args:
            documents: List of document dicts with 'id', 'title', 'text', 'category'.
            
        Returns:
            self
        """
        if not documents:
            raise ValueError("Cannot index empty document collection.")

        self.documents = list(documents)
        cleaned_texts = [
            self.cleaner.clean(f"{doc.get('title', '')} {doc.get('text', '')}")
            for doc in self.documents
        ]

        self.doc_matrix = self.tfidf.fit_transform(cleaned_texts)
        self.is_indexed = True
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """Compute similarity scores for all indexed documents against a query.
        
        Args:
            query: Query string.
            
        Returns:
            1D numpy array of similarity scores.
        """
        if not self.is_indexed or self.doc_matrix is None:
            raise RuntimeError("Engine has not been indexed yet.")

        cleaned_query = self.cleaner.clean(query)
        if not cleaned_query.strip():
            return np.zeros(len(self.documents), dtype=np.float32)

        query_vec = self.tfidf.transform(cleaned_query)[0]
        return compute_similarities(query_vec, self.doc_matrix)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search indexed documents for a query and return ranked Top-K results.
        
        Args:
            query: User query string.
            top_k: Number of results to retrieve.
            
        Returns:
            List of ranked result dictionaries.
        """
        scores = self.get_scores(query)
        return rank_top_k(scores, self.documents, top_k=top_k)
