"""
Hybrid retrieval engine combining BM25 keyword search and semantic dense vector similarity.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np

from app.chunking.chunker import Chunk
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.top_k import reciprocal_rank_fusion


class HybridRetriever:
    """
    Blends keyword (BM25) and semantic retrieval scores:
        Score_hybrid = alpha * Norm(Score_semantic) + (1 - alpha) * Norm(Score_keyword)
    where alpha in [0.0, 1.0] controls the balance between semantic and lexical relevance.
    """
    def __init__(
        self,
        keyword_retriever: KeywordRetriever,
        semantic_retriever: SemanticRetriever,
        alpha: float = 0.5
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self.keyword_retriever = keyword_retriever
        self.semantic_retriever = semantic_retriever
        self.alpha = alpha

    def _minmax_normalize(self, scored_items: List[Tuple[Chunk, float]]) -> Dict[str, float]:
        """Normalizes scores to [0.0, 1.0] range."""
        if not scored_items:
            return {}
        scores = [s for _, s in scored_items]
        min_s = min(scores)
        max_s = max(scores)
        denom = max_s - min_s

        norm_dict = {}
        for chunk, s in scored_items:
            norm_dict[chunk.metadata.chunk_id] = (s - min_s) / denom if denom > 1e-9 else 1.0
        return norm_dict

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        min_score: float = 0.0,
        category_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Executes parallel retrieval across keyword and semantic engines,
        normalizes scores, merges with weight alpha, and returns ranked top_k.
        """
        weight = alpha if alpha is not None else self.alpha
        candidate_k = max(top_k * 3, 20)

        # 1. Retrieve candidates
        kw_results = self.keyword_retriever.retrieve(
            query, top_k=candidate_k, min_score=0.0, category_filter=category_filter
        )
        sem_results = self.semantic_retriever.retrieve(
            query, top_k=candidate_k, min_score=0.0, category_filter=category_filter
        )

        if not kw_results and not sem_results:
            return []

        # 2. Normalize scores
        kw_norm = self._minmax_normalize(kw_results)
        sem_norm = self._minmax_normalize(sem_results)

        # 3. Collect all candidate chunks
        all_chunks: Dict[str, Chunk] = {}
        for c, _ in kw_results:
            all_chunks[c.metadata.chunk_id] = c
        for c, _ in sem_results:
            all_chunks[c.metadata.chunk_id] = c

        # 4. Compute blended score
        blended: List[Tuple[Chunk, float]] = []
        for cid, chunk in all_chunks.items():
            k_score = kw_norm.get(cid, 0.0)
            s_score = sem_norm.get(cid, 0.0)
            final_score = (weight * s_score) + ((1.0 - weight) * k_score)

            if final_score >= min_score:
                blended.append((chunk, float(final_score)))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:top_k]
