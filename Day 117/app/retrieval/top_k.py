"""
Query result representation, top-k filtering, and reciprocal rank fusion utilities.
"""
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

from app.chunking.chunker import Chunk


@dataclass
class QueryResult:
    """Represents a retrieved chunk with ranking metadata and confidence score."""
    chunk: Chunk
    score: float
    rank: int
    retrieval_method: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.metadata.chunk_id,
            "document_id": self.chunk.metadata.document_id,
            "title": self.chunk.metadata.title,
            "category": self.chunk.metadata.category,
            "score": round(self.score, 4),
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
            "text": self.chunk.text
        }


def format_retrieval_results(
    raw_results: List[Tuple[Chunk, float]],
    method: str = "semantic"
) -> List[QueryResult]:
    """Wraps raw (Chunk, score) tuples into indexed QueryResult objects."""
    formatted = []
    for rank, (chunk, score) in enumerate(raw_results, start=1):
        formatted.append(QueryResult(
            chunk=chunk,
            score=score,
            rank=rank,
            retrieval_method=method
        ))
    return formatted


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Chunk, float]]],
    k_constant: int = 60
) -> List[Tuple[Chunk, float]]:
    """
    RRF algorithm:
        RRF_Score(d) = sum_{m in Models} 1 / (k_constant + rank_m(d))
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for r_list in ranked_lists:
        for rank, (chunk, _) in enumerate(r_list, start=1):
            cid = chunk.metadata.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)

    merged = [(chunk_map[cid], score) for cid, score in rrf_scores.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged
