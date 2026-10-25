"""
Hybrid Retrieval Engine.
Combines sparse keyword matching (BM25) and dense semantic embeddings using
weighted score normalization and Reciprocal Rank Fusion (RRF).
"""
from typing import List, Optional, Callable, Dict
import numpy as np

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalizes a dictionary of scores to [0.0, 1.0]."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v = min(vals)
    max_v = max(vals)
    diff = max_v - min_v
    if diff == 0.0:
        return {k: 1.0 if max_v > 0 else 0.0 for k in scores}
    return {k: (v - min_v) / diff for k, v in scores.items()}


class HybridRetriever:
    """
    Hybrid search engine combining BM25 keyword matching and dense semantic retrieval.
    Supports both weighted score fusion and Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        candidate_k: int = 20,
        rrf_k0: int = 60
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha  # weight for dense; (1 - alpha) for bm25
        self.candidate_k = candidate_k
        self.rrf_k0 = rrf_k0

    def search_weighted(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        filter_fn: Optional[Callable[[Chunk], bool]] = None
    ) -> List[ScoredChunk]:
        """
        Executes hybrid retrieval using Min-Max normalized score fusion:
            score = alpha * norm(dense_score) + (1 - alpha) * norm(bm25_score)
        """
        eff_alpha = self.alpha if alpha is None else alpha
        if not (0.0 <= eff_alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {eff_alpha}")

        # Fetch candidate sets from both retrievers
        k_candidates = max(top_k * 2, self.candidate_k)
        bm25_results = self.bm25.search(query, top_k=k_candidates, filter_fn=filter_fn)
        dense_results = self.dense.search(query, top_k=k_candidates, filter_fn=filter_fn)

        if not bm25_results and not dense_results:
            return []

        # Build chunk map
        chunk_map: Dict[str, Chunk] = {}
        bm25_scores: Dict[str, float] = {}
        dense_scores: Dict[str, float] = {}

        for sc in bm25_results:
            cid = sc.chunk.metadata.chunk_id
            chunk_map[cid] = sc.chunk
            bm25_scores[cid] = sc.score

        for sc in dense_results:
            cid = sc.chunk.metadata.chunk_id
            chunk_map[cid] = sc.chunk
            dense_scores[cid] = sc.score

        # Normalize scores across candidate union
        all_ids = set(chunk_map.keys())
        for cid in all_ids:
            if cid not in bm25_scores:
                bm25_scores[cid] = 0.0
            if cid not in dense_scores:
                dense_scores[cid] = 0.0

        norm_bm25 = min_max_normalize(bm25_scores)
        norm_dense = min_max_normalize(dense_scores)

        # Compute fused score
        fused: List[ScoredChunk] = []
        for cid, chunk in chunk_map.items():
            s_dense = norm_dense.get(cid, 0.0)
            s_bm25 = norm_bm25.get(cid, 0.0)
            fused_score = eff_alpha * s_dense + (1.0 - eff_alpha) * s_bm25
            fused.append(ScoredChunk(chunk=chunk, score=float(fused_score)))

        fused.sort(key=lambda sc: (sc.score, sc.chunk.metadata.chunk_id), reverse=True)
        return fused[:top_k]

    def search_rrf(
        self,
        query: str,
        top_k: int = 5,
        k0: Optional[int] = None,
        filter_fn: Optional[Callable[[Chunk], bool]] = None
    ) -> List[ScoredChunk]:
        """
        Executes hybrid retrieval using Reciprocal Rank Fusion:
            RRF(d) = sum_{r in {bm25, dense}} 1 / (k0 + rank_r(d))
        """
        eff_k0 = self.rrf_k0 if k0 is None else k0
        k_candidates = max(top_k * 2, self.candidate_k)

        bm25_results = self.bm25.search(query, top_k=k_candidates, filter_fn=filter_fn)
        dense_results = self.dense.search(query, top_k=k_candidates, filter_fn=filter_fn)

        chunk_map: Dict[str, Chunk] = {}
        rrf_scores: Dict[str, float] = {}

        # 1-based ranks
        for rank, sc in enumerate(bm25_results, start=1):
            cid = sc.chunk.metadata.chunk_id
            chunk_map[cid] = sc.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (eff_k0 + rank)

        for rank, sc in enumerate(dense_results, start=1):
            cid = sc.chunk.metadata.chunk_id
            chunk_map[cid] = sc.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (eff_k0 + rank)

        fused: List[ScoredChunk] = []
        for cid, score in rrf_scores.items():
            fused.append(ScoredChunk(chunk=chunk_map[cid], score=float(score)))

        fused.sort(key=lambda sc: (sc.score, sc.chunk.metadata.chunk_id), reverse=True)
        return fused[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "rrf",
        filter_fn: Optional[Callable[[Chunk], bool]] = None
    ) -> List[ScoredChunk]:
        """Unified search interface supporting 'rrf' or 'weighted'."""
        if method.lower() == "weighted":
            return self.search_weighted(query, top_k=top_k, filter_fn=filter_fn)
        return self.search_rrf(query, top_k=top_k, filter_fn=filter_fn)
