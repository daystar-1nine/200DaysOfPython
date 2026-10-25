"""
Error Taxonomy and Failure Mode Analysis for Advanced RAG.
Categorizes retrieval failures into lexical mismatch, semantic confusion, redundancy displacement, and rank decay.
"""
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Sequence
from collections import Counter


@dataclass
class QueryErrorDiagnosis:
    """Diagnostic breakdown for a single benchmark query evaluation."""
    query_id: str
    query: str
    query_type: str
    target_docs: List[str]
    retrieved_docs: List[str]
    error_mode: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type,
            "target_docs": self.target_docs,
            "retrieved_docs": self.retrieved_docs,
            "error_mode": self.error_mode,
            "explanation": self.explanation
        }


def diagnose_retrieval_error(
    query: str,
    query_id: str,
    query_type: str,
    target_doc_ids: Set[str],
    retrieved_doc_ids: List[str],
    bm25_top_ids: List[str],
    dense_top_ids: List[str],
    k: int = 5
) -> QueryErrorDiagnosis:
    """
    Diagnoses failure mode by contrasting BM25, Dense, and final fused retrieval rankings.
    """
    top_k_retrieved = retrieved_doc_ids[:k]
    is_hit = any(doc in target_doc_ids for doc in top_k_retrieved)

    if not target_doc_ids:
        # Unanswerable query
        if not top_k_retrieved:
            return QueryErrorDiagnosis(
                query_id=query_id,
                query=query,
                query_type=query_type,
                target_docs=[],
                retrieved_docs=top_k_retrieved,
                error_mode="correct_rejection",
                explanation="Successfully identified unanswerable query."
            )
        else:
            return QueryErrorDiagnosis(
                query_id=query_id,
                query=query,
                query_type=query_type,
                target_docs=[],
                retrieved_docs=top_k_retrieved,
                error_mode="hallucination_risk",
                explanation="Retrieved spurious documents for an unanswerable query."
            )

    if is_hit:
        return QueryErrorDiagnosis(
            query_id=query_id,
            query=query,
            query_type=query_type,
            target_docs=list(target_doc_ids),
            retrieved_docs=top_k_retrieved,
            error_mode="correct_retrieval",
            explanation="Target document successfully ranked in top-K."
        )

    # Check why it missed
    bm25_hit = any(doc in target_doc_ids for doc in bm25_top_ids[:k])
    dense_hit = any(doc in target_doc_ids for doc in dense_top_ids[:k])

    if dense_hit and not bm25_hit:
        error_mode = "lexical_mismatch"
        explanation = "Dense search found relevant chunk; BM25 failed due to vocabulary mismatch."
    elif bm25_hit and not dense_hit:
        error_mode = "semantic_confusion"
        explanation = "BM25 keyword search found exact identifier; Dense search was misled by embedding space proximity."
    elif any(doc in target_doc_ids for doc in retrieved_doc_ids):
        # Found deeper in list
        error_mode = "rank_decay"
        rank = next(idx + 1 for idx, d in enumerate(retrieved_doc_ids) if d in target_doc_ids)
        explanation = f"Target document retrieved at rank {rank}, but displaced outside top-{k}."
    else:
        error_mode = "candidate_miss"
        explanation = "Target document completely missing from candidate pool."

    return QueryErrorDiagnosis(
        query_id=query_id,
        query=query,
        query_type=query_type,
        target_docs=list(target_doc_ids),
        retrieved_docs=top_k_retrieved,
        error_mode=error_mode,
        explanation=explanation
    )


class ErrorAnalyzer:
    """Aggregates and reports error taxonomy distributions across benchmark queries."""
    def __init__(self):
        self.diagnoses: List[QueryErrorDiagnosis] = []

    def record(self, diagnosis: QueryErrorDiagnosis) -> None:
        self.diagnoses.append(diagnosis)

    def summarize_distribution(self) -> Dict[str, int]:
        counts = Counter(d.error_mode for d in self.diagnoses)
        return dict(counts)
