"""
Semantic dense vector retrieval engine.
"""
from typing import List, Tuple, Optional
import numpy as np

from app.chunking.chunker import Chunk
from app.embeddings.embedder import BaseEmbedder
from app.index.vector_index import VectorIndex


class SemanticRetriever:
    """
    Semantic retrieval engine projecting queries to embedding space
    and querying a VectorIndex for cosine-nearest chunks.
    """
    def __init__(self, embedder: BaseEmbedder, index: VectorIndex):
        self.embedder = embedder
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        category_filter: Optional[str] = None,
        doc_id_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Embeds the query and queries the vector index.
        """
        if not query or not query.strip():
            return []

        if self.index.size() == 0:
            return []

        q_vec = self.embedder.embed_text(query)
        return self.index.search(
            query_vector=q_vec,
            top_k=top_k,
            min_score=min_score,
            category_filter=category_filter,
            doc_id_filter=doc_id_filter
        )
