"""
Reciprocal Rank Fusion (RRF) Engine.
Combines multiple ranked lists from heterogeneous retrieval systems into a single unified ranking.
Based on Cormack, Clarke, and Büttcher (SIGIR 2009).
"""
from typing import List, Dict, Optional, Tuple, Sequence
from collections import defaultdict

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    k0: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_k: Optional[int] = None
) -> List[ScoredChunk]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion:
        RRF_score(d) = sum_{r in R} w_r / (k0 + rank_r(d))
    
    Args:
        ranked_lists: Sequence of ranked lists of ScoredChunks (1-based ranking).
        k0: Smoothing constant (standard default = 60). Mitigates high-rank outlier dominance.
        weights: Optional sequence of weights w_r for each ranking system. Defaults to 1.0 for all.
        top_k: If provided, truncates the final fused list to top_k elements.
        
    Returns:
        List of ScoredChunks sorted descending by fused RRF score.
    """
    if not ranked_lists:
        return []

    if k0 < 0:
        raise ValueError(f"k0 must be non-negative, got {k0}")

    num_lists = len(ranked_lists)
    eff_weights = list(weights) if weights is not None else [1.0] * num_lists
    if len(eff_weights) != num_lists:
        raise ValueError(f"Length of weights ({len(eff_weights)}) does not match ranked_lists ({num_lists})")

    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_store: Dict[str, Chunk] = {}
    best_rank_seen: Dict[str, int] = {}

    for list_idx, rank_list in enumerate(ranked_lists):
        w = eff_weights[list_idx]
        for rank_1based, scored_chunk in enumerate(rank_list, start=1):
            chunk = scored_chunk.chunk
            cid = chunk.metadata.chunk_id
            
            chunk_store[cid] = chunk
            rrf_contribution = w / (k0 + rank_1based)
            rrf_scores[cid] += rrf_contribution
            
            if cid not in best_rank_seen or rank_1based < best_rank_seen[cid]:
                best_rank_seen[cid] = rank_1based

    # Sort descending by RRF score; break ties by best individual rank then chunk_id
    fused: List[ScoredChunk] = [
        ScoredChunk(chunk=chunk_store[cid], score=score)
        for cid, score in rrf_scores.items()
    ]
    fused.sort(
        key=lambda sc: (
            sc.score,
            -best_rank_seen.get(sc.chunk.metadata.chunk_id, 999999),
            sc.chunk.metadata.chunk_id
        ),
        reverse=True
    )

    if top_k is not None:
        return fused[:top_k]
    return fused


class ReciprocalRankFusion:
    """
    Stateful helper for Reciprocal Rank Fusion.
    """
    def __init__(self, k0: int = 60):
        self.k0 = k0

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[ScoredChunk]],
        weights: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None
    ) -> List[ScoredChunk]:
        """Fuses multiple ranked lists using RRF."""
        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k0=self.k0,
            weights=weights,
            top_k=top_k
        )
