"""
Dense Vector Retrieval Engine.
Embeds user queries into dense metric spaces and performs top-K cosine similarity retrieval over VectorIndex.
"""
from typing import List, Optional, Callable
import numpy as np

from app.chunking.chunker import Chunk
from app.embeddings.embedder import BaseEmbedder
from app.index.vector_index import VectorIndex, ScoredChunk


class DenseRetriever:
    """
    Dense semantic retriever searching over dense vector embeddings.
    """
    def __init__(self, embedder: BaseEmbedder, index: VectorIndex):
        self.embedder = embedder
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
        min_score: float = -1.0
    ) -> List[ScoredChunk]:
        """Embeds query and searches VectorIndex."""
        if not query or not query.strip():
            return []

        if len(self.index) == 0:
            return []

        q_vec = self.embedder.embed_text(query)
        return self.index.search(
            query_vector=q_vec,
            top_k=top_k,
            category=category,
            filter_fn=filter_fn,
            min_score=min_score
        )
