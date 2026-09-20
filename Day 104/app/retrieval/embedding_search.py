"""
Embedding Search Engine module for Day 104 Semantic Search Engine.
Implements dense semantic retrieval using document embeddings and cosine similarity.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from ..preprocessing.cleaner import TextCleaner
from ..preprocessing.tokenizer import Tokenizer
from ..representations.document_embedding import DocumentEmbedder
from .ranking import compute_similarities, rank_top_k


class EmbeddingSearchEngine:
    """Semantic search engine utilizing dense document embeddings and cosine similarity."""

    def __init__(
        self,
        embedder: Optional[DocumentEmbedder] = None,
        cleaner: Optional[TextCleaner] = None,
        tokenizer: Optional[Tokenizer] = None,
        pooling: str = "mean"
    ):
        self.embedder = embedder if embedder is not None else DocumentEmbedder()
        self.cleaner = cleaner if cleaner is not None else TextCleaner()
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.pooling = pooling
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_matrix: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def index(
        self,
        documents: List[Dict[str, Any]],
        tfidf_model: Any = None,
        train_embeddings: bool = True
    ) -> "EmbeddingSearchEngine":
        """Index documents by computing dense document embeddings.
        
        Args:
            documents: List of document dictionaries.
            tfidf_model: Optional fitted TfidfRepresentation for weighted pooling.
            train_embeddings: Whether to train word embeddings on this corpus if not trained.
            
        Returns:
            self
        """
        if not documents:
            raise ValueError("Cannot index empty document collection.")

        self.documents = list(documents)
        self.tokenized_corpus = []

        for doc in self.documents:
            full_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            cleaned = self.cleaner.clean(full_text)
            tokens = self.tokenizer.tokenize(cleaned)
            self.tokenized_corpus.append(tokens)

        if train_embeddings and not self.embedder.embeddings:
            self.embedder.train_on_corpus(self.tokenized_corpus)

        self.doc_matrix = self.embedder.embed_corpus(
            self.tokenized_corpus,
            pooling=self.pooling,
            tfidf_model=tfidf_model
        )
        self.is_indexed = True
        return self

    def get_scores(self, query: str, tfidf_model: Any = None) -> np.ndarray:
        """Compute semantic similarity scores for all documents against a query.
        
        Args:
            query: Query string.
            tfidf_model: Optional TF-IDF model for weighted query pooling.
            
        Returns:
            1D numpy array of similarity scores.
        """
        if not self.is_indexed or self.doc_matrix is None:
            raise RuntimeError("Engine has not been indexed yet.")

        cleaned = self.cleaner.clean(query)
        tokens = self.tokenizer.tokenize(cleaned)

        if not tokens:
            return np.zeros(len(self.documents), dtype=np.float32)

        weights = None
        if self.pooling == "weighted" and tfidf_model is not None:
            weights = tfidf_model.get_token_weights(cleaned)

        query_vec = self.embedder.embed_document(tokens, pooling=self.pooling, weights=weights)
        return compute_similarities(query_vec, self.doc_matrix)

    def search(self, query: str, top_k: int = 5, tfidf_model: Any = None) -> List[Dict[str, Any]]:
        """Search indexed documents for a query and return ranked Top-K results.
        
        Args:
            query: User query string.
            top_k: Number of results to retrieve.
            tfidf_model: Optional TF-IDF model for weighted query pooling.
            
        Returns:
            List of ranked result dictionaries.
        """
        scores = self.get_scores(query, tfidf_model=tfidf_model)
        return rank_top_k(scores, self.documents, top_k=top_k)
