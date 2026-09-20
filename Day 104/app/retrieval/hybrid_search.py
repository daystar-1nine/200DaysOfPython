"""
Hybrid Search Engine module for Day 104 Semantic Search Engine.
Combines TF-IDF keyword scores and dense embedding similarity scores with normalization.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .ranking import normalize_scores, rank_top_k
from .tfidf_search import TfidfSearchEngine
from .embedding_search import EmbeddingSearchEngine


class HybridSearchEngine:
    """Combines TF-IDF keyword retrieval and dense semantic search via convex interpolation."""

    def __init__(
        self,
        tfidf_engine: TfidfSearchEngine,
        embedding_engine: EmbeddingSearchEngine,
        default_alpha: float = 0.5
    ):
        self.tfidf_engine = tfidf_engine
        self.embedding_engine = embedding_engine
        self.default_alpha = default_alpha

    def get_scores(self, query: str, alpha: Optional[float] = None) -> np.ndarray:
        """Compute normalized hybrid similarity scores for all documents.
        
        Formula:
            Score = alpha * norm(TFIDFScore) + (1 - alpha) * norm(EmbeddingScore)
            
        Args:
            query: Query string.
            alpha: Weight for TF-IDF component (in [0, 1]).
            
        Returns:
            1D numpy array of combined hybrid scores.
        """
        w = self.default_alpha if alpha is None else alpha
        w = max(0.0, min(1.0, float(w)))

        tfidf_raw = self.tfidf_engine.get_scores(query)
        emb_raw = self.embedding_engine.get_scores(query)

        tfidf_norm = normalize_scores(tfidf_raw)
        emb_norm = normalize_scores(emb_raw)

        return (w * tfidf_norm + (1.0 - w) * emb_norm).astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search indexed documents using hybrid scoring.
        
        Args:
            query: User query string.
            top_k: Number of results to retrieve.
            alpha: Weight for TF-IDF component.
            
        Returns:
            List of ranked result dictionaries with individual and combined scores.
        """
        w = self.default_alpha if alpha is None else alpha
        w = max(0.0, min(1.0, float(w)))

        tfidf_raw = self.tfidf_engine.get_scores(query)
        emb_raw = self.embedding_engine.get_scores(query)

        tfidf_norm = normalize_scores(tfidf_raw)
        emb_norm = normalize_scores(emb_raw)

        hybrid_scores = (w * tfidf_norm + (1.0 - w) * emb_norm).astype(np.float32)
        base_results = rank_top_k(hybrid_scores, self.tfidf_engine.documents, top_k=top_k)

        # Enrich results with component scores
        doc_id_to_idx = {doc["id"]: i for i, doc in enumerate(self.tfidf_engine.documents)}
        for item in base_results:
            idx = doc_id_to_idx[item["document_id"]]
            item["tfidf_score"] = float(tfidf_raw[idx])
            item["embedding_score"] = float(emb_raw[idx])
            item["hybrid_score"] = float(hybrid_scores[idx])

        return base_results
