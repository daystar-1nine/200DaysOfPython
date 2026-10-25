"""
Evaluation package for Information Retrieval metrics, latency profiling, and error analysis.
"""
from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    discounted_cumulative_gain_at_k,
    ndcg_at_k,
    evaluate_ranking,
)
from app.evaluation.latency import LatencyTracker
from app.evaluation.error_analysis import (
    QueryErrorDiagnosis,
    diagnose_retrieval_error,
    ErrorAnalyzer,
)
from app.evaluation.grounding import (
    compute_citation_precision,
    evaluate_faithfulness,
)
from app.evaluation.evaluator import RAGEvaluator

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "discounted_cumulative_gain_at_k",
    "ndcg_at_k",
    "evaluate_ranking",
    "LatencyTracker",
    "QueryErrorDiagnosis",
    "diagnose_retrieval_error",
    "ErrorAnalyzer",
    "compute_citation_precision",
    "evaluate_faithfulness",
    "RAGEvaluator",
]
